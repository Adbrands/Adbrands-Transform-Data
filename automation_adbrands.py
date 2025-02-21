import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import json
import re
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

def update_company_map_if_needed(master, company_map_df, credentials_file, spreadsheet_url, batch_size=50, max_writes_per_minute=90):
    """
    Efficiently update the company map sheet by batching updates, handling API limits per minute, and retrying when needed.
    """

    master['Company'] = master['Company'].replace('', np.nan)
    master['Brand'] = master['Brand'].replace('', np.nan)
    
    # Find new unmatched companies
    unmatched_companies = master[['OLD BRAND NAME', 'Company', 'Brand']].drop_duplicates()
    unmatched_companies = unmatched_companies[~unmatched_companies['OLD BRAND NAME'].isin(company_map_df['OLD BRAND NAME'])]

    if unmatched_companies.empty:
        st.info("ℹ️ **No New Company Entries to Update.**")
        return company_map_df

    st.info(f"**New Company Entries Found:** {len(unmatched_companies)} records")

    # Fill missing values
    unmatched_companies['Company'] = unmatched_companies['Company'].fillna('Unknown')
    unmatched_companies['Brand'] = unmatched_companies['Brand'].fillna('Unknown')

    # Google Sheets API setup
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
    client = gspread.authorize(credentials)
    sheet = client.open_by_url(spreadsheet_url).sheet1

    # Convert data to a list format suitable for batch appending
    rows_to_add = unmatched_companies[['OLD BRAND NAME', 'Company', 'Brand']].values.tolist()

    # Batch processing with per-minute rate control
    total_batches = (len(rows_to_add) // batch_size) + 1
    writes_used = 0

    for i in range(total_batches):
        batch = rows_to_add[i * batch_size : (i + 1) * batch_size]
        
        if batch:
            try:
                sheet.append_rows(batch)
                st.success(f"✅ **Batch {i+1}/{total_batches} Added Successfully!**")
                writes_used += len(batch)

                # If we reach the per-minute limit, pause for a minute
                if writes_used >= max_writes_per_minute:
                    st.warning(f"⏳ **API limit reached. Waiting 60 seconds before continuing...**")
                    time.sleep(60)
                    writes_used = 0  # Reset counter after waiting
                
            except Exception as e:
                st.error(f"⚠️ **API Error: {e} - Retrying in 10 seconds...**")
                time.sleep(10)
                sheet.append_rows(batch)  # Retry once

    # Update the local company map dataframe
    company_map_df = pd.concat([company_map_df, unmatched_companies], ignore_index=True).drop_duplicates(subset=['OLD BRAND NAME'])
    
    st.success("🎉 **All New Company Entries Successfully Added!**")
    st.markdown("---")
    return company_map_df
    
def update_market_url_if_needed(new_data, market_url_df, credentials_file, spreadsheet_url):
    """
    Update the market_url sheet with new 'Territory' values if they don't exist,
    ensuring no duplicates (including permutations or subsets).
    """
    def normalize_territory(territory):
        return ', '.join(sorted([x.strip() for x in territory.split(',')]))

    market_url_df['Normalized Territory'] = market_url_df['Territory'].apply(normalize_territory)
    existing_territories = set(market_url_df['Normalized Territory'])

    new_data['Normalized Territory'] = new_data['Territory'].apply(normalize_territory)
    new_territories = set(new_data['Normalized Territory'].unique()) - existing_territories

    if new_territories:
        st.info("**New Territories Found:**")
        st.write(f"- {', '.join(new_territories)}")

        duplicates = []
        for new_territory in new_territories:
            new_countries = set(new_territory.split(', '))
            for existing_territory in existing_territories:
                existing_countries = set(existing_territory.split(', '))
                if new_countries == existing_countries or new_countries.issubset(existing_countries) or existing_countries.issubset(new_countries):
                    duplicates.append(new_territory)
                    break
        
        new_territories_to_append = new_territories - set(duplicates)

        if new_territories_to_append:
            st.success("**Territories to Append:**")
            st.write(f"- {', '.join(new_territories_to_append)}")

            new_rows = pd.DataFrame({
                'Territory': list(new_territories_to_append),
                'Region': ['Unknown'] * len(new_territories_to_append)
            })

            SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            credentials = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
            client = gspread.authorize(credentials)
            sheet = client.open_by_url(spreadsheet_url).sheet1

            for _, row in new_rows.iterrows():
                sheet.append_row([row['Territory'], row['Region']])

            st.success("**Territories Successfully Added to Google Sheet.**")
            new_rows['Normalized Territory'] = new_rows['Territory']
            market_url_df = pd.concat([market_url_df, new_rows], ignore_index=True)
        else:
            st.warning("⚠️ **No Unique Territories to Append. All are duplicates.**")
    else:
        st.info("ℹ️ **No New Territories to Update.**")

    market_url_df = market_url_df.drop_duplicates(subset=['Normalized Territory'])
    market_url_df = market_url_df.drop(columns=['Normalized Territory'])
    st.markdown("---")
    return market_url_df

def update_agency_map_if_needed(master, agency_map_df, credentials_file, spreadsheet_url):
    """
    Update the agency map sheet with new 'Agency Description' values if no 'Current Agency MATCH' exists.
    """
    unmatched_agencies = master[master['Current Agency MATCH'].isna()]['Current Agency Description'].unique()
    unmatched_agencies = set(unmatched_agencies) - set(agency_map_df['Agency Description'])

    if unmatched_agencies:
        st.info("**New Agencies Found:**")
        st.write(f"- {', '.join(unmatched_agencies)}")
        new_rows = pd.DataFrame({'Agency Description': list(unmatched_agencies), 'Match': ['Unknown'] * len(unmatched_agencies)})

        SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
        client = gspread.authorize(credentials)
        sheet = client.open_by_url(spreadsheet_url).sheet1

        for _, row in new_rows.iterrows():
            sheet.append_row([row['Agency Description'], row['Match']])

        st.success("**New Agencies Successfully Added to Google Sheet.**")
        agency_map_df = pd.concat([agency_map_df, new_rows], ignore_index=True)
    else:
        st.info("ℹ️ **No New Agencies to Update.**")

    agency_map_df = agency_map_df.drop_duplicates(subset=['Agency Description'])
    st.markdown("---")
    return agency_map_df


def convert_df_to_excel(df):
    output = BytesIO()  # Create a BytesIO buffer
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Processed Data')
    processed_data = output.getvalue()  # Retrieve the file content
    return processed_data

def authenticate_and_load_data(credentials_file, spreadsheet_url, sheet_name="Sheet1"):
    """ ss """
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

    credentials = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
    client = gspread.authorize(credentials)
    sheet = client.open_by_url(spreadsheet_url).worksheet(sheet_name)
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

def map_market_to_region(market, territories_dict):
    """Maps each market to the right region by looping through the territories dictionary."""
    market_countries = set([x.strip() for x in market.split(',')])
    
    # Check for exact matches
    for key in territories_dict:
        key_countries = set([x.strip() for x in key.split(',')])
        if market_countries == key_countries:
            return territories_dict[key]
    
    # If no exact match, check for subset matches
    for key in territories_dict:
        key_countries = set([x.strip() for x in key.split(',')])
        if market_countries.issubset(key_countries):
            return territories_dict[key]
    
    # Return None if no match found
    return None

def clean_territory(value):
    """  ss """
    if "South Asia" in value:
        return "South Asia (incl. Bangladesh, India, Pakistan, Sri Lanka)"
    if 'Perú' in value:
        return "Peru"
    if 'Panamá' in value:
        return 'Panama'
    value = re.sub(r'\b(\w+)\.(\w+)\b', r'\1\2', value)
    value = re.sub(r'\.(?=\s|$)', '', value)
    value = re.sub(r'\b(Including|including|and|&|the|The)\b', '', value)
    value = re.sub(r'&', '', value)
    value = re.sub(r'\[.*?\]|\(.*?\)', '', value)
    value = value.replace(';', ',')
    value = re.sub(r'[.,;:!?]$', '', value)
    return re.sub(r'\s+', ' ', value).strip()

def read_excel_upload(media_creative, latest_data, company_brand, agency_matches):
    """Reads and processes uploaded Excel files with proper duplicate handling."""
    if media_creative is None:
        st.error("Media/Creative file not uploaded.")
        return None, None
    
    # Determine file type based on file name
    file_name = media_creative.name
    file_type = 'Media' if 'Media' in file_name else 'Creative'
    
    try:
        # Read the Excel file and clean column names
        data = pd.read_excel(media_creative, sheet_name=file_type + ' Wins', header=7)
        data.columns = data.columns.str.strip().str.replace(r'\s+', ' ', regex=True).str.replace(' ', '')

        # Filter data based on file type
        if file_type == 'Creative':
            data = data[data['YY/N'] == 'N']
        elif file_type == 'Media':
            data = data[data['ConfidentialY/N'] == 'N']

        # Filter for specific month
        data = data[data['Month'] == selected_month]

        # Clean and process columns
        data['Client'] = data['Client'].str.strip()
        data['Agency1'] = data['Agency']
        data['Agency'] = data['Agency'].str.strip().str.title()
        data['Status'] = 'Awarded'
        data['Market'] = data['Market'].fillna('').astype(str)
        data['Remark'] = data['Remark'].fillna('').astype(str)
        data['Type of Assignment'] = np.where(
            data['Remark'] == '',
            'AOR/Project',
            'AOR/Project - ' + data['Remark']
        )
        data['Current Agency Description'] = data['Agency']
        data['Incumbent Agency Description'] = data['Incumbent']
        data['Assignment'] = file_type
        # Deduplicate the company_brand table
        company_brand_cleaned = company_brand.drop_duplicates(subset=['OLD BRAND NAME'])

        # Merge with company_brand
        data = data.merge(
            company_brand_cleaned, 
            how='left', 
            left_on='Client', 
            right_on='OLD BRAND NAME'
        )

        # Clean a copy of agency_matches to prevent in-place modification
        agency_matches_cleaned = agency_matches[['Agency Description', 'Match']].drop_duplicates()
        agency_matches_cleaned['Agency Description'] = agency_matches_cleaned['Agency Description'].str.strip().str.title()

        # Merge for Incumbent MATCH
        data = data.merge(
            agency_matches_cleaned, 
            how='left', 
            left_on='Incumbent Agency Description', 
            right_on='Agency Description'
        )
        data = data.rename(columns={'Match': 'Incumbent MATCH'}).drop(columns=['Agency Description'])

        # Merge for Current Agency MATCH
        data['Current Agency Description'] = data['Current Agency Description'].str.strip().str.title()
        data = data.merge(
            agency_matches_cleaned, 
            how='left', 
            left_on='Current Agency Description', 
            right_on='Agency Description'
        )
        data = data.rename(columns={'Match': 'Current Agency MATCH'}).drop(columns=['Agency Description'])
    
        # Add additional fields
        data['Territory'] = data['Market'].apply(clean_territory).replace(territory_mapping)
        data['Market'] = data['Market'].apply(clean_territory).replace(territory_mapping)

        # Convert Month column
        data['Month'] = pd.to_datetime(data['Month'] + ' 2024', format='%b %Y', errors='coerce')
        data['Month'] = data['Month'].dt.strftime('%d/%m/%Y')

        # Format Billings
        data['Est Billings'] = data['Billings(US$k)'].apply(lambda x: f"USD${x * 1_000:,.0f}")
        data['OLD BRAND NAME'] = data['Client']
        data['Current Agency Description'] = data['Agency1']
    except Exception as e:
        st.error(f"Error processing Media/Creative file: {e}")
        return None, None

    # Process latest_data (master sheet)
    try:
        excel_file = pd.ExcelFile(latest_data)
        if 'Main - NEW FORMULAS' in excel_file.sheet_names:
            master_sheet = pd.read_excel(latest_data, sheet_name='Main - NEW FORMULAS')
        else:
            master_sheet = pd.read_excel(latest_data)

        master_sheet['Date'] = pd.to_datetime(master_sheet['Date'], dayfirst=True, errors='coerce')
        master_sheet['Date'] = master_sheet['Date'].dt.strftime('%d/%m/%Y')
    
    except Exception as e:
        st.error(f"Error processing Data Mastersheet file: {e}")
        return None, None

    return data, master_sheet, file_type

st.title('Adbrands Automation')
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

selected_month = st.selectbox(
    "Month",
    MONTHS,
    index=8,  # Default to September
    help="Choose the month for which you want to filter the data to transformed."
)

# Display the selected month
st.success(f"You have selected: **{selected_month}**")
st.markdown("---")
territory_mapping = {
    "United Arab Emirates": "UAE",
    "Korea": "South Korea",
    "US": "USA",
    "Saudi": "Saudi Arabia"
}

url = "https://drive.google.com/uc?export=download&id=1spEQnYeoVTcjgpJzg4OYySXA4kEW1FEQ"
response = requests.get(url)

# Save the JSON locally
with open("credentials.json", "wb") as f:
    f.write(response.content)

# Update the path
credentials_url = "credentials.json"

market_url = authenticate_and_load_data(credentials_url, "https://docs.google.com/spreadsheets/d/1viHyvUYXb4HmdyEhCy4vLqOB2pRiMbFlt8ciSnLfmTw/edit?pli=1&gid=0#gid=0")
agency_url = authenticate_and_load_data(credentials_url, "https://docs.google.com/spreadsheets/d/1LZpxcmmJAQhYWfwfj1mHOnnfM__8omXIbS38ug6lk6I/edit?gid=0#gid=0")
company_url = authenticate_and_load_data(credentials_url, "https://docs.google.com/spreadsheets/d/1cnbAqbVvd2S_glID662w55lgWp3u6r2Pm3rb3HIPXDY/edit?gid=0#gid=0")

col1, col2 = st.columns(2)

with col1:
    uploaded_excel = st.file_uploader(
        "Raw Data File",
        type=["xlsx", "xlsm"],
        help="Upload the latest Media or Creative data file in .xlsx or .xlsm format.",
    )

with col2:
    uploaded_excel2 = st.file_uploader(
        "Data Mastersheet File",
        type=["xlsx", "xlsm"],
        help="Upload the latest Data Mastersheet file in .xlsx or .xlsm format.",
    )

# Only proceed if both files are uploaded
if uploaded_excel is not None and uploaded_excel2 is not None:
    if st.button("Click to start"):
        st.markdown("---")
        # Read and process the uploaded files
        data, master_sheet, file_types = read_excel_upload(uploaded_excel, uploaded_excel2, company_url, agency_url)

        # Transform and clean the data
        master = data[['Month', 'OLD BRAND NAME', 'Company', 'Brand', 'Status', 'Assignment', 
                    'Type of Assignment', 'Territory', 'Region', 'Current Agency MATCH', 
                    'Current Agency Description', 'Incumbent MATCH', 'Incumbent Agency Description', 
                    'Category_y', 'Est Billings']]

        master = master.rename(columns={
            'Category_y': 'Categories Updated', 
            'Month': 'Date', 
            'Incumbent Agency Description': 'Incumbant Agency Description'
        })

        market_url = update_market_url_if_needed(data, market_url, credentials_url,
                                                "https://docs.google.com/spreadsheets/d/1viHyvUYXb4HmdyEhCy4vLqOB2pRiMbFlt8ciSnLfmTw/edit?pli=1&gid=0")
        agency_url = update_agency_map_if_needed(master, agency_url, credentials_url,
                                                "https://docs.google.com/spreadsheets/d/1LZpxcmmJAQhYWfwfj1mHOnnfM__8omXIbS38ug6lk6I")
        company_url = update_company_map_if_needed(master, company_url, credentials_url,
                                                "https://docs.google.com/spreadsheets/d/1cnbAqbVvd2S_glID662w55lgWp3u6r2Pm3rb3HIPXDY/edit?gid=0#gid=0")

        # Recompute territories dictionary
        territories = dict(zip(market_url['Territory'], market_url['Region']))
        master['Region'] = data['Market'].apply(lambda x: map_market_to_region(x, territories))
        master['Date'] = pd.to_datetime(master['Date'])
        master['Date'] = master['Date'].dt.strftime('%d/%m/%Y')
        columns_to_clean = ['Company', 'Brand', 'Current Agency MATCH']
        master[columns_to_clean] = master[columns_to_clean].replace('Unknown', None)
        current_date = datetime.now()
        formatted_date = current_date.strftime('%m.%d.%y')
        file_name = f"{formatted_date} {file_types}_Sheet.xlsx"


        last_filled_row = master_sheet.shape[0]
        combined_data = pd.concat([master_sheet, master], ignore_index=True)
        num_transformed_data_rows = len(master)
        num_mastersheet_rows = len(combined_data)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Last Filled Row", value=last_filled_row)
        with col2:
            st.metric(label="Transformed Data Rows", value=num_transformed_data_rows)
        with col3:
            st.metric(label="Mastersheet Rows", value=num_mastersheet_rows)
        st.markdown("---")
        with st.container():
            st.markdown("### Transformed Data")

            with st.expander("Click to View Transformed Data"):
                st.dataframe(master)

            excel_file = convert_df_to_excel(master)
            
            st.download_button(
                label="Download Transformed Data as Excel",
                data=excel_file,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("⚠️ **Please click the button to start.**")
