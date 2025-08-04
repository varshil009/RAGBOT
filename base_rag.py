import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import io

# for tokenizers and reading pdf
from transformers import AutoTokenizer, AutoModel
import torch
import fitz  # PyMuPDF
import pdfplumber

# Display the variable in Markdown format
#from IPython.display import Markdown, display

# for api
import requests
import json

# to calculate similarities
from sklearn.metrics.pairwise import cosine_similarity

class pdf_processing:
    def __init__(self, path, chunk_size):
        self.path = path
        self.chunk_size = chunk_size
        # saves all text in one var and other saves the text page wise
        self.text, self.text_p = "", []
        self.chunks = []

    def extract_text_and_chunks(self):
        # extract text
        """with fitz.open(self.path) as pdf:
            for page in pdf:
                self.text_p.append(page.get_text())
                self.text += page.get_text()
        return self.text_p, self.text"""

        """# pypdf2 works well with some other forms of pdfs as well
        with open(self.path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pages = reader.pages
            total_pages = len(pages)

            for i in range(total_pages):
                textn = pages[i].extract_text()
                self.text_p.append(textn)
                self.text += textn
        """
        pdf = pdfplumber.open(self.path)
        print("==> extract text and chunks started")

        textwise_page = {}
        for i, page in enumerate(pdf.pages):
            text = page.extract_text().split()
            for j in range(0, len(text), 100):
                new_chunk = " ".join(text[j : j + self.chunk_size]).lower()
                textwise_page[new_chunk] = i

        text_chunks =  np.array(list(textwise_page.keys()))
        print("==> extract text and chunks finished")

        return text_chunks, textwise_page

        """
        
        def create_chunks(self):
        # create chunks
        for i in range(0, len(self.text), self.chunk_size):
            new_chunk = self.text[i : i + self.chunk_size].lower()
            self.chunks.append(new_chunk)
        return self.chunks
        """

    def extract_captions_and_images(self):
        print("==> extract captions and images started")

        def check_caption(cap):
            k = cap.lower()
            if "figure" in k: pass
            elif "fig" in k: pass
            elif "chart" in k: pass
            else:
                return False

            return True

        images = {}

        with fitz.open(self.path) as fitz_pdf:
            for page_num, page in enumerate(fitz_pdf):
                #print(f"Processing Page {page_num + 1}")

                # First get the raw image list
                image_list = page.get_images()
                if image_list:
                    #print(f"Page {page_num + 1} contains {len(image_list)} image(s).")

                    # Get locations of images on page
                    img_info_list = page.get_image_info()

                    # Extract text blocks with their bounding boxes
                    text_blocks = page.get_text("words")

                    # Process each image
                    for img_index, img in enumerate(image_list):
                        # Get the image location from img_info_list
                        img_info = img_info_list[img_index]
                        image_bbox = img_info['bbox']
                        #print(f"Image bounding box: {image_bbox}")

                        # Extract the raw image data
                        base_image = fitz_pdf.extract_image(img[0])
                        image_bytes = base_image["image"]

                        # Find description text below image
                        descr = ""
                        for block in text_blocks:
                            if block[1] > image_bbox[3] and block[1] < image_bbox[3] + 20:
                                descr += " " + block[4]

                        if check_caption(descr):
                            pass

                        else:
                            descr = ""
                            for block in text_blocks:
                                if block[1] < image_bbox[3] and block[1] > image_bbox[1]:
                                    descr += " " + block[4]

                            if check_caption(descr):
                                pass
                            else:

                                descr = ""
                                for block in text_blocks:
                                    if block[1] < image_bbox[1] and block[1] > image_bbox[1] + 20:
                                        descr += " " + block[4]

                        # Convert to PIL Image while maintaining original quality
                        images[descr] = [Image.open(io.BytesIO(image_bytes)), image_bbox]

        print("==> extract captions and images finished")
        return images

    

class embedd:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        #model_path = r"C:\Users\Varshil\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        print("==> model ready to work")

    def generate_embeddings(self, chunks):
        embeddings = []
        print("==> started processing the chunks")
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors = 'pt', truncation = True, padding = True)
            #print("inputs : ", inputs)
            """
            The tokenizer processes the text chunk and converts it into a format suitable for the model.
            # args...

            return_tensors='pt': This argument specifies that the output should be in PyTorch tensor format, which is required for the model.
            truncation=True: This ensures that any input longer than the model's maximum length is truncated, preventing errors during processing.
            padding=True: This ensures that shorter inputs are padded to the same length, allowing for batch processing.

            # keys that are returned and which will be used as arg to model:
            input_ids : list of token ids of all tokenised words
            attention_mask : binary mask indicating which tokes are to be attended by the model
            token_type_ids :  It indicates which tokens belong to which segment, if all tokens belong to a single segment then [0,0,0,0]
            overflowing_tokens : This key contains any tokens that were truncated when the input exceeded the maximum length allowed by the model.
            num_truncated_tokens : number of truncated tokesm
            """
            with torch.no_grad():
                outputs = self.model(**inputs)
                """
                No Gradient Calculation: The with torch.no_grad(): context manager is used to disable gradient calculations. This is important during inference to save memory
                and speed up computations since we don't need gradients for backpropagation.
                Model Output: The model processes the tokenized inputs and returns the outputs, which include various hidden states.
                The **inputs syntax unpacks the dictionary of input tensors into keyword arguments(as described earlier) for the model.
                """

                k = outputs.last_hidden_state
                #print("meaned last hidden layer : ", k.shape) # prints mean of all multidimensional layers
                embeddings.append(k.mean(dim=1).squeeze().numpy())
                # last hidden state is output of last layer
                """
                Extracting Last Hidden State:
                outputs.last_hidden_state contains the hidden states for all tokens in the input sequence. This is a tensor of shape (batch_size, sequence_length, hidden_size).
                Mean Calculation:
                mean(dim=1) computes the mean of the hidden states across all tokens in the sequence, effectively creating a single embedding for the entire input chunk.
                This is done to obtain a fixed-size vector representation for each chunk.
                Squeeze and Convert to NumPy:
                    """
        print("==> finished processing the chunks")
    
        return embeddings

    def search(self, query, embeddings, chunks, chunk_page):
        query_embedding = self.generate_embeddings([query])[0]
        similarities = cosine_similarity([query_embedding], embeddings)
        best_match_index = similarities.argsort()[0, ::-1]
        ref = [chunks[best_match_index[x]] for x in range(5)]
        page_nums = [chunk_page[x] for x in ref]
        return "\n".join(ref), page_nums

    def fetch_image(self, query, cap_embeddings, cap_img):
        print("==> fetching image")
        query_embedding = self.generate_embeddings([query])[0]
        similarities = cosine_similarity([query_embedding], cap_embeddings)
        similarities = np.where(similarities >= 0.4, similarities, -1)
        best_match_index = similarities.argsort()[0, ::-1][0]
        img = cap_img[list(cap_img.keys())[best_match_index]][0]
        print("==> image fetched successfully")
        return img

class LLM:
  def __init__(self):
    print("==> LLM initiating")
    self.key = ""
    self.url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.key}'
    self.headers = {'Content-Type': 'application/json'}
    self.query = None
    self.rag_response = None
    print("==> LLM initiated")

  def _prompt_dic(self, prompt):
    return {
              "contents": [
                  {
                      "parts": [
                          {
                              "text": prompt
                          }
                      ]
                  }
              ],

              "generationConfig": {
                  "temperature": 0.7,
                  "topK": 40,
                  "topP": 0.95,
                  "maxOutputTokens": 1024,
              }
            }

  def generate_rag(self, query):
    self.query = query
    prompt = f"""
                            You are an AI assistant capable of processing large documents and providing detailed, structured responses.
                            Your task is to analyze the user query and guide a retrieval system to fetch relevant information from a knowledge base or document repository.

                            Here’s the workflow:
                            1. I will provide you with a query or a goal.
                            2. Analyze the query and list the key information, topics, or concepts that should be retrieved to answer it.

                            ### Input Query:
                            {self.query}

                            ### Your Output:
                            1. Identify key information or topics relevant to the query.
                            2. Suggest search terms or filters to retrieve the most relevant content.

                            """

    response = requests.post(self.url, headers = self.headers, json = self._prompt_dic(prompt))
    r = response.json()
    r = r['candidates'][0]['content']['parts'][0]['text']
    print("==> enhanced query generated")

    return r

  def generate_final_response(self, rag_response):

    prompt = f"""You are a chatbot geared with RAG. you are provided reference information from RAG mechanism
                Here is the retrieved information. Refine your response by integrating this data to provide a complete answer to the query.

                ### Original Query:
                {self.query}

                ### Retrieved Information:
                {rag_response}

                ### Your Task:
                1. Synthesize the retrieved data into a coherent, detailed response.
                2. Present the final response in a user-friendly format, highlighting key points and providing structured details if required.
                Return answer as if you are interacting with user.
              """
    print("==> sending enhanced query")
    response = requests.post(self.url, headers = self.headers, json = self._prompt_dic(prompt))
    r = response.json()
    r = r['candidates'][0]['content']['parts'][0]['text']
    print("==> final response received")

    return r


class rag_process:
    def __init__(self, path):
        print("==> Processing of pdf started")
        self.path = path
        obj = pdf_processing(self.path, 200)
        #print("chunking...")
        self.chunks, self.chunk_page = obj.extract_text_and_chunks()
        #print("extracting images...")
        self.cap_img = obj.extract_captions_and_images()
        self.em = embedd()
        self.embeddings = self.em.generate_embeddings(self.chunks)
        self.cap_embeddings = self.em.generate_embeddings(list(self.cap_img.keys()))
        print("==> Processing the pdf finished")

    def execute(self, query):
        #path = "/content/Deep Learning with Python.pdf"
        
        """"""
        # this query comes from backend in which it comes from frontend
        query = query.lower()
        llm = LLM()
        enhanced_query = llm.generate_rag(query)
        rag_response, page_nums = self.em.search(enhanced_query, self.embeddings, self.chunks, self.chunk_page)
        final_response = llm.generate_final_response(rag_response)
        bin_img = self.em.fetch_image(query, self.cap_embeddings, self.cap_img)
        print("the page nums", page_nums)
        return final_response, bin_img, page_nums