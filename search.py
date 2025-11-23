import openai
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

# Set up OpenAI client for LM Studio local server
client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"  # LM Studio doesn't require a key
)

# Function to generate embedding using Nomic Embed Text V2
def get_embedding(text, prefix="search_document: "):
    response = client.embeddings.create(
        model="local-model",  # LM Studio uses a generic name; adjust if needed
        input=prefix + text
    )
    embedding = response.data[0].embedding
    return embedding[:256]  # Use MRL to reduce to 256 dims for efficiency

# Define a Pydantic model for the table schema
class Document(LanceModel):
    id: int = Field()
    text: str = Field()
    vector: Vector(256) = Field()  # Vector dimension matches MRL choice

# Set up LanceDB
db_path = "./lancedb"  # Local directory for the DB
db = lancedb.connect(db_path)
table_name = "semantic_search"

# Create table if it doesn't exist
if table_name not in db.table_names():
    db.create_table(table_name, schema=Document)

table = db.open_table(table_name)

# Example documents to index
documents = [
    {"id": 1, "text": "The quick brown fox jumps over the lazy dog."},
    {"id": 2, "text": "Semantic search uses vector embeddings for meaning-based retrieval."},
    {"id": 3, "text": "Nomic Embed Text V2 supports multilingual tasks efficiently."},
    {"id": 4, "text": "Artificial intelligence is transforming industries worldwide."},
    {"id": 5, "text": "Machine learning models require high-quality training data."},
    {"id": 6, "text": "বাংলা ভাষা আমাদের গর্ব এবং সংস্কৃতির প্রতীক।"},
    {"id": 7, "text": "আজকের আবহাওয়া অনেক সুন্দর, আকাশে হালকা মেঘ আছে।"},
    {"id": 8, "text": "Data-driven decision making helps companies grow faster."},
    {"id": 9, "text": "আমি প্রতিদিন নতুন কিছু শিখতে পছন্দ করি।"},
    {"id": 10, "text": "Cloud computing makes it easier to scale applications globally."},
    {"id": 11, "text": "বিজ্ঞান ও প্রযুক্তির অগ্রগতিতে মানুষের জীবনমান উন্নত হয়েছে।"},
    {"id": 12, "text": "Real-time translation bridges communication gaps between cultures."},
    {"id": 13, "text": "একটি স্মার্ট চ্যাটবট গ্রাহক সাপোর্টের মান উন্নত করতে সাহায্য করে।"},
    {"id": 14, "text": "Python is widely used in data science and AI development."},
    {"id": 15, "text": "বাংলাদেশ দক্ষিণ এশিয়ার একটি সুন্দর দেশ।"},
    {"id": 16, "text": "Efficient indexing improves the performance of search engines."},
    {"id": 17, "text": "আমরা আজ একটি নতুন মেশিন লার্নিং মডেল ট্রেন করেছি।"},
    {"id": 18, "text": "Neural networks are inspired by the human brain’s structure."},
    {"id": 19, "text": "বৃষ্টির দিনে এক কাপ চা আর বই পড়া দারুণ লাগে।"},
    {"id": 20, "text": "Large language models can understand and generate multiple languages."},
    {"id": 21, "text": "Bangladesh is known for its rivers, greenery, and hospitality."},
    {"id": 22, "text": "The capital of Bangladesh is Dhaka, a vibrant and busy city."},
    {"id": 23, "text": "Japan is famous for its cherry blossoms and advanced technology."},
    {"id": 24, "text": "India has a rich cultural heritage and diverse languages."},
    {"id": 25, "text": "The United States is a global leader in innovation and research."},
    {"id": 26, "text": "Switzerland is known for its mountains, chocolates, and banking system."},
    {"id": 27, "text": "France attracts millions of tourists every year for its art and cuisine."},
    {"id": 28, "text": "Australia is home to kangaroos and the Great Barrier Reef."},
    {"id": 29, "text": "Brazil is famous for football and the Amazon rainforest."},
    {"id": 30, "text": "Canada is known for its polite people and stunning natural landscapes."},
    {"id": 31, "text": "চীন বিশ্বের সবচেয়ে জনবহুল দেশ এবং এর প্রাচীন ইতিহাস অত্যন্ত সমৃদ্ধ।"},
    {"id": 32, "text": "জাপান প্রযুক্তিতে অত্যন্ত উন্নত এবং এর সংস্কৃতি বিশ্বজুড়ে জনপ্রিয়।"},
    {"id": 33, "text": "আমেরিকা একটি বহুজাতিক সমাজ যেখানে বিভিন্ন সংস্কৃতির মানুষ একসাথে বাস করে।"},
    {"id": 34, "text": "ভারত তার ঐতিহ্য, খাবার এবং উৎসবের জন্য বিখ্যাত।"},
    {"id": 35, "text": "ইংল্যান্ডের রাজধানী লন্ডন, যা একটি ঐতিহাসিক এবং আধুনিক শহর।"},
    {"id": 36, "text": "Germany is known for its engineering excellence and Oktoberfest celebration."},
    {"id": 37, "text": "Saudi Arabia is the birthplace of Islam and home to the holy cities Mecca and Medina."},
    {"id": 38, "text": "Italy is famous for its art, architecture, and delicious cuisine."},
    {"id": 39, "text": "South Korea has become a global hub for technology and K-pop culture."},
    {"id": 40, "text": "Russia is the largest country in the world by land area."},
    {"id": 41, "text": "বাংলাদেশের সুন্দরবন পৃথিবীর সবচেয়ে বড় ম্যানগ্রোভ বন।"},
    {"id": 42, "text": "মালদ্বীপ তার নীল সমুদ্র ও রিসোর্টের জন্য বিখ্যাত।"},
    {"id": 43, "text": "নেপাল হিমালয় পর্বতমালার দেশ, যেখানে পৃথিবীর সর্বোচ্চ শৃঙ্গ এভারেস্ট অবস্থিত।"},
    {"id": 44, "text": "Sri Lanka is an island nation known for tea and tropical beaches."},
    {"id": 45, "text": "The Netherlands is famous for windmills, tulips, and canals."},
    {"id": 46, "text": "Egypt is known for its pyramids and the ancient Nile civilization."},
    {"id": 47, "text": "Norway has breathtaking fjords and one of the highest standards of living."},
    {"id": 48, "text": "Singapore is a small but powerful economy in Southeast Asia."},
    {"id": 49, "text": "Turkey connects Europe and Asia both geographically and culturally."},
    {"id": 50, "text": "South Africa is known for its wildlife and diverse population."},
    {"id": 51, "text": "বাংলাদেশের পতাকা সবুজ রঙের যার মধ্যে লাল বৃত্ত দেশের স্বাধীনতার প্রতীক।"},
    {"id": 52, "text": "ভুটান একটি শান্তিপ্রিয় দেশ যা ‘Gross National Happiness’-এর ধারণা অনুসরণ করে।"},
    {"id": 53, "text": "United Kingdom consists of England, Scotland, Wales, and Northern Ireland."},
    {"id": 54, "text": "Indonesia is the world’s largest archipelago with thousands of islands."},
    {"id": 55, "text": "The economy of China has grown rapidly over the last few decades."},
    {"id": 56, "text": "Bangladesh exports garments, textiles, and leather goods globally."},
    {"id": 57, "text": "বাংলাদেশের কক্সবাজার বিশ্বের সবচেয়ে দীর্ঘ সমুদ্র সৈকতগুলোর একটি।"},
    {"id": 58, "text": "France is also known for the Eiffel Tower and romantic ambiance of Paris."},
    {"id": 59, "text": "Spain is famous for flamenco dance and Mediterranean cuisine."},
    {"id": 60, "text": "Canada shares the longest international border with the United States."}
]


# Generate embeddings and index documents
data_to_add = []
for doc in documents:
    embedding = get_embedding(doc["text"])
    data_to_add.append(Document(id=doc["id"], text=doc["text"], vector=embedding))

table.add(data_to_add)
print("Documents indexed successfully.")

# Perform semantic search (cosine similarity by default)
query_text = "slothful animal"
query_embedding = get_embedding(query_text, prefix="search_query: ")

# Search for top 2 results
results = table.search(query_embedding).limit(2).to_list()

# Display results
for result in results:
    print(f"Score: {result['_distance']:.4f} - Text: {result['text']}")