import tkinter as tk
from tkinter import ttk, scrolledtext
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class AnkiCardCreator:
    def __init__(self, master):
        self.master = master
        master.title("Anki Card Creator")

        # LLM selector
        ttk.Label(master, text="Select LLM:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.llm_selector = ttk.Combobox(master, values=[
            "GPT-4o mini", "GPT-4o", "GPT-4 Turbo",
            "Claude 3 Opus", "Claude 3.5 Sonnet", "Claude 3 Haiku"
        ])
        self.llm_selector.set("GPT-4o mini")
        self.llm_selector.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # Input text area
        ttk.Label(master, text="Input Text:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.text_input = scrolledtext.ScrolledText(master, height=10)
        self.text_input.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Create cards button
        self.create_button = ttk.Button(master, text="Create Cards", command=self.create_cards)
        self.create_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Output area
        self.output_area = scrolledtext.ScrolledText(master, height=10)
        self.output_area.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Configure grid
        master.columnconfigure(1, weight=1)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(4, weight=1)

    def create_cards(self):
        text = self.text_input.get("1.0", tk.END).strip()
        llm = self.llm_selector.get()
        
        self.output_area.delete("1.0", tk.END)
        self.output_area.insert(tk.END, "Creating cards...\n")
        
        try:
            cards = self.create_anki_cards(text, llm)
            self.output_area.insert(tk.END, f"Created cards:\n{cards}")
        except Exception as e:
            self.output_area.insert(tk.END, f"Error: {str(e)}")

    def create_anki_cards(self, text: str, llm: str) -> str:
        client = ChatOpenAI()
        system_template = "Create Anki cards from the given text."
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "{text}")
        ])
        chain = prompt_template | client | StrOutputParser()
        return chain.invoke({"text": text})

if __name__ == "__main__":
    root = tk.Tk()
    app = AnkiCardCreator(root)
    root.mainloop()
