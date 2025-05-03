## RAG Q/A Chatbot


# Project Overview 
This project involves scraping Historical documents from Wikipedia, extracting relevant details, scrape the data and storing the scrapped data, embedding it into a vector database (Weaviate), and using a retrieval-augmented generation (RAG) pipeline with a chatbot for querying the stored information. Additionally, the chatbot responses are evaluated using various metrics like rouge_score, cosine_similarity, bert_score_f1 and other parameters like faithfulness, answer_relevancy, context_precision, context_recall.



# 1. Web Scrapping


## 📌 Features

- Scrapes introductory content from Wikipedia pages.
- Supports retry logic with customizable delay.
- Cleans and filters paragraph text.
- Logs scraping and saving activities to a log file.
- Stores all data in a specified directory.

---

## 🛠️ Technologies Used

* **Python 3**
* `requests` – HTTP requests to fetch web content
* `BeautifulSoup` – HTML parsing and data extraction
* `os`, `re`, `time` – File handling, regex parsing, and delays

---

## ⚙️ Workflow

1. **Fetch Wikipedia Page**
   The script makes a request to the Wikipedia page listing

2. **Extract Table Data**
   Locates the correct table by identifying headers that contain topics.

4. **Scrape Specific Pages**

   * Extracts and cleans full-text content by removing unwanted whitespace and references
   * Saves the content as a `.txt` file named

---

## 🧩 Key Functions

### `@staticmethod clean_text(text)`
Cleans a given text by:
- Removing extra spaces.
- Joining text into a single line for consistent formatting.

---

### `scrape_page(self, title, retries=3, delay=2)`
Fetches and parses the main content of a given Wikipedia page.
- Retries the request up to `retries` times if it fails.
- Waits for `delay` seconds between retries to prevent overload.

---

### `save_to_file(self, file_path, content)`
Saves the cleaned content into a file:
- Encoded in UTF-8.
- Creates parent directories if they do not exist.

---

### `scrape_all(self)`
Runs the complete scraping process:
- Iterates through all provided topics.
- Scrapes each corresponding Wikipedia page.
- Appends results and saves them to a final text file.

---


# 2. Creating chunking and Embeddings of data and storing it into WEAVIATE 



# 3. RAG Bot



# 4. Evaluation








## Conclusion 
This project successfully integrates web scraping, vector search, and LLM-based chatbot functionalities to provide an interactive knowledge retrieval system. By leveraging Weaviate for efficient storage and retrieval, along with an evaluation pipeline, the system ensures high-quality responses with measurable accuracy.












<!-- ### `get_unicorn_startups()`

* Extracts startup details and links from the Wikipedia unicorn list page.
* Returns a list of dictionaries, one per startup.

### `scrape_startup_page(startup, index)`

* Visits the startup’s Wikipedia page using the link.
* Parses, cleans, and saves the content to a `.txt` file.

### `scrape_unicorns()`

* Driver function that coordinates:

  * Table scraping
  * Iterating over startups
  * Company page scraping and file saving

--- -->
