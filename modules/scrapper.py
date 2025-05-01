


import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import os

TOPICS = [
    
    "Indus_Valley_Civilisation",
    "Vedic_period",
    "Mahajanapadas",
    "Maurya_Empire",
    "Ashoka",
    "Gupta_Empire",
    "Nalanda",
    "Chalukya_dynasty",
    "Pallava_dynasty",
    "Rashtrakuta_dynasty",
    "Satavahana_dynasty",
    "Sangam_period",
    "Kushan_Empire",
    "Shunga_Empire",
    "Kanva_dynasty",
    "Magadha",
    "Kalinga_(historical_region)",
    "Mathura",
    "Takshashila",
    "Pataliputra",
    "Ajanta_Caves",
    "Ellora_Caves",
    "Bharhut",
    "Sanchi",
    "Bodh_Gaya",
    "Sarnath",
    "Lumbini",
    "Vaishali",
    "Rajgir",
    "Ujjain",
    "Ayodhya",
    "Hastinapur",
    "Kurukshetra",
    "Dholavira",
    "Mohenjo-daro",
    "Harappa",
    "Rakhigarhi",
    "Lothal",
    "Kalibangan",
    "Mehrgarh",
    "Bimbisara",
    "Ajatashatru",
    "Chanakya",
    "Chandragupta_Maurya",
    "Bindusara",
    "Samudragupta",
    "Chandragupta_I",
    "Chandragupta_II",
    "Harsha",
    "Pushyamitra_Shunga",
    "Delhi_Sultanate",
    "Mughal_Empire",
    "Babur",
    "Humayun",
    "Akbar",
    "Jahangir",
    "Shah_Jahan",
    "Aurangzeb",
    "Maratha_Empire",
    "Shivaji",
    "Peshwa",
    "Rajput",
    "Rana_Prathap",
    "Battle_of_Haldighati",
    "Vijayanagara_Empire",
    "Krishnadevaraya",
    "Bahmani_Sultanate",
    "Deccan_sultanates",
    "Qutb_Shahi_dynasty",
    "Golconda",
    "Bidar",
    "Bijapur_Sultanate",
    "Ahmadnagar_Sultanate",
    "Berar_Sultanate",
    "Khandesh_Sultanate",
    "Mewar",
    "Chittorgarh",
    "Jaisalmer",
    "Bikaner",
    "Amber",
    "Jaipur",
    "Jodhpur",
    "Udaipur",
    "Gwalior",
    "Bundelkhand",
    "Malwa_Sultanate",
    "Gondwana",
    "Ahom_dynasty",
    "Koch_dynasty",
    "Tripura_Kingdom",
    "Manipur_Kingdom",
    "Chola_dynasty",
    "Raja_Raja_Chola_I",
    "Rajendra_Chola_I",
    "Pandyas",
    "Cheras",
    "Hoysalas",
    "Kakatiya_dynasty",
    "Yadava_dynasty",
    "Paramara_dynasty",
    "British_East_India_Company",
    "Battle_of_Plassey",
    "Battle_of_Buxar",
    "Warren_Hastings",
    "Lord_Cornwallis",
    "Lord_Wellesley",
    "Lord_Dalhousie",
    "Doctrine_of_Lapse",
    "Indian_Rebellion_of_1857",
    "Rani_Lakshmibai",
    "Mangal_Pandey",
    "Bahadur_Shah_II",
    "Nana_Sahib",
    "Tatya_Tope",
    "Begum_Hazrat_Mahal",
    "British_Raj",
    "Indian_National_Congress",
    "Partition_of_Bengal_(1905)",
    "Swadeshi_movement",
    "Home_Rule_Movement",
    "Rowlatt_Act",
    "Jallianwala_Bagh_massacre",
    "Non-cooperation_movement",
    "Salt_March",
    "Civil_Disobedience_Movement",
    "Quit_India_Movement",
    "Indian_Independence_Act_1947",
    "Partition_of_India",
    "Mountbatten_Plan",
    "Cripps_Mission",
    "Cabinet_Mission",
    "Simon_Commission",
    "Government_of_India_Act_1935",
    "Round_Table_Conferences",
    "Indian_Civil_Service",
    "Viceroy_of_India",
    "Indian_Army_during_British_rule",
    "Indian_Police_Service",
    "Indian_Foreign_Service",
    "Indian_Education_Service",
    "Indian_Medical_Service",
    "Indian_Postal_Service",
    "Indian_Railways",
    "Indian_Telegraph_Service",
    "Indian_Currency_during_British_rule",
    "Indian_States_and_territories_during_British_rule",
    "Princely_states_of_India",
    "Chamber_of_Princes",
    "Indian_Political_Service",
    "Indian_Independence_movement",
    "Constitution_of_India",
    "First_general_election_in_India",
    "States_Reorganisation_Act_1956",
    "Green_Revolution_in_India",
    "White_Revolution_in_India",
    "Operation_Blue_Star",
    "Emergency_(India)",
    "Indo-Pakistani_War_of_1947–1948",
    "Indo-Pakistani_War_of_1965",
    "Indo-Pakistani_War_of_1971",
    "Kargil_War",
    "Indian_general_election,_2014",
    "Indian_general_election,_2019",
    "Ayodhya_dispute",
    "Babri_Masjid",
    "Ram_Janmabhoomi",
    "Article_370_of_the_Constitution_of_India",
    "Abrogation_of_Article_370",
    "Demonetisation_in_India",
    "Goods_and_Services_Tax_(India)",
    "Make_in_India",
    "Digital_India",
    "Jan_Dhan_Yojana",
    "Swachh_Bharat_Abhiyan",
    "COVID-19_pandemic_in_India",
    "Farmers'_Protest_(India,_2020–2021)",
    "Women_in_India",
    "LGBT_rights_in_India",
    "Education_in_India",
    "History_of_Indian_currency",
    "NITI_Aayog",
    "Planning_Commission_(India)",
    "Five-Year_Plans_of_India",
    "Indian_space_programme",
    "ISRO",
    "Chandrayaan-1",
    "Chandrayaan-2",
    "Chandrayaan-3",
    "Mangalyaan",
    "Indian_Navy",
    "Indian_Air_Force",
    "India_and_the_Non-Aligned_Movement",
    "India–United_States_relations",
    "India–Russia_relations",
    "India–China_relations",
    "India–Pakistan_relations",
    "SAARC",
    "BRICS",
    "G20",
    "Nehruvian_socialism",
    "License_Raj",
    "Economic_liberalisation_in_India",
    "History_of_the_Indian_rupee"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

def clean_text(text):
    return ' '.join(text.split())

def scrape_wikipedia_page(title, retries=3, delay=2):
    url = f"https://en.wikipedia.org/wiki/{title}"
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                print(f"⚠️ Failed to fetch {title} (Status code: {response.status_code})")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            content_div = soup.find("div", {"class": "mw-parser-output"})
            if not content_div:
                print(f"⚠️ Content div not found for {title}")
                continue

            paragraphs = content_div.find_all("p")
            text = "\n".join(clean_text(p.get_text()) for p in paragraphs if len(p.get_text(strip=True)) > 50)
            return text

        except Exception as e:
            print(f"❌ Error scraping {title} (Attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    return ""

def save_data(topics):
    full_text = ""
    for topic in tqdm(topics, desc="Scraping Wikipedia Topics"):
        print(f"🔍 Scraping: {topic}")
        text = scrape_wikipedia_page(topic)
        if text:
            full_text += f"\n\n---\n\n# {topic.replace('_', ' ')}\n\n{text}"
        else:
            print(f"🚫 No content fetched for: {topic}")
        time.sleep(1.5)  # Be gentle on Wikipedia

    file_path = os.path.join(SAVE_DIR, "modern_history_of_india.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_text.strip())
    print(f"\n✅ Data saved to: {file_path}")

if __name__ == "__main__":
    save_data(TOPICS)