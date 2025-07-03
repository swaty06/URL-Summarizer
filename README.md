# 🔎 URL Summarizer (RAG + ChromaDB)

A lightweight Retrieval-Augmented Generation (RAG) app that summarizes content from any web page and answers user questions — using ChromaDB for vector storage and a powerful LLM for summarization and Q&A.

---

## 🚀 Features

- **🌐 Summarize Any URL:** Paste 1–3 URLs, and the tool fetches, processes, and summarizes content in seconds.
- **💬 Intelligent Q&A:** Ask questions like *“What is this article about?”* or *“What are the key takeaways?”* — the model retrieves relevant content and responds accurately.
- **📚 Multi-Source Support:** Input multiple URLs to get a broader and more comprehensive view of a topic.
- **⚙️ RAG Architecture:** Combines retrieval (ChromaDB vector store) with generation (LLM) for accurate, context-aware outputs.
- **🧠 LLM-Backed Reasoning:** Uses a language model to synthesize and summarize even long and unstructured text.
- **🔧 Simple & Fast:** Designed for non-technical users — just paste, process, and interact.

---

## 🧠 How RAG Works in This Tool

1. **Input:** User provides 1–3 URLs.
2. **Data Extraction:** The app scrapes and processes raw content from each link.
3. **Embedding & Storage:** Content is split, embedded using an embedding model, and stored in **ChromaDB**.
4. **Retrieval:** When the user asks a question, relevant content is retrieved based on semantic similarity.
5. **Answer Generation:** An LLM generates the final response using the retrieved chunks as context.

---

## 🎯 Use Cases

- **🧾 Article Summarization:** Instantly condense long-form content from blogs, news sites, or technical documentation.
- **💡 Knowledge Extraction:** Ask deep questions and get direct, context-aware answers from multiple sources.
- **📚 Learning Aid:** Use it to study multiple resources without reading them line by line.
- **🧪 Research Assistant:** Quickly gather, synthesize, and query information from various domains.

---

## 🔮 Future Enhancements

- **➕ Unlimited URLs:** Extend support for more than 3 URLs.
- **🌐 Browser Plugin:** Enable summarization directly from any browser tab.
- **🧠 Domain-Tuned RAG:** Fine-tune the model for specific content types (e.g., legal, academic, product reviews).
- **📥 File Upload:** Allow uploading PDFs or text files in addition to URLs.

---

## 🚀 Live Demo

👉 Try the app here: [URL Summarizer (RAG)](https://url-summarizer-emhbqm88xs2hojxymwy7gw.streamlit.app/)

---

## 🛠️ How to Use

1. Open the [Live App](https://url-summarizer-emhbqm88xs2hojxymwy7gw.streamlit.app/)
2. Paste up to 3 URLs in the input fields.
3. Click **“Process URLs”** to extract and store data.
4. Ask your custom question — the app will fetch relevant content and generate an answer.

---

## 📦 Tech Stack

- **🧠 Language Model:** OpenAI / Hugging Face LLM
- **📥 Data Loader:** `UnstructuredURLLoader` for content scraping
- **🧮 Embeddings:** OpenAI or HuggingFace embedding model
- **🧾 Vector DB:** ChromaDB
- **🖥️ UI:** Streamlit

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Contributions Welcome

Feel free to fork, improve, or suggest enhancements via pull requests or issues!

