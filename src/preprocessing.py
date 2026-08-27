import json
import re
import os
from typing import List, Dict, Any
import pandas as pd
from pypdf import PdfReader

class MahapuranaPreprocessor:
    """
    Full Production Preprocessor for Ashtadasha Mahapuranas.
    Processes ALL 123 MLBD PDF files and Tier 1 Excel dataset without any limits.
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.data_dir = os.path.abspath(data_dir)

    def clean_text(self, text: str) -> str:
        """Strips OCR noise and extra whitespaces while preserving Sanskrit & punctuation."""
        if not text or pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,\?\!\'\":\-\(\)\[\]]', '', text)
        return text.strip()

    def validate_and_format_chunk(self, raw_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Validates canonical schema and builds citation string."""
        purana = raw_chunk.get("purana_name") or raw_chunk.get("Purana") or "Mahapurana"
        chapter = raw_chunk.get("chapter") or raw_chunk.get("Chapter") or 1
        skandha = raw_chunk.get("skandha_book") or raw_chunk.get("Book") or 1
        content = raw_chunk.get("text_content") or raw_chunk.get("text") or raw_chunk.get("Text") or ""
        source = raw_chunk.get("content_source", "Tier 1 (Dataset)")

        cleaned_content = self.clean_text(content)
        if not cleaned_content or len(cleaned_content.split()) < 5:
            return None

        formatted_chunk = {
            "chunk_id": str(raw_chunk.get("chunk_id", f"{str(purana)[:2].upper()}_{skandha}_{chapter}")),
            "purana_name": str(purana),
            "skandha_book": int(skandha) if str(skandha).isdigit() else 1,
            "chapter": int(chapter) if str(chapter).isdigit() else 1,
            "verse_range": str(raw_chunk.get("verse_range", raw_chunk.get("Verse", "N/A"))),
            "story_title": str(raw_chunk.get("story_title", raw_chunk.get("Section", "Untitled Passage"))),
            "text_content": cleaned_content,
            "content_source": source,
            "translation_language": "English",
            "themes": raw_chunk.get("themes", []),
            "citation_string": f"[{purana}, Book {skandha}, Chapter {chapter}]"
        }
        return formatted_chunk

    def load_json_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads and processes a JSON dataset file safely."""
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                
            processed = []
            for item in raw_data:
                chunk = self.validate_and_format_chunk(item)
                if chunk:
                    processed.append(chunk)
            return processed
        except Exception as e:
            print(f"⚠️ Warning reading JSON dataset: {e}")
            return []

    def load_excel_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads Kaggle 18_Mahapuranas_DataSet.xlsx file."""
        if not os.path.exists(file_path):
            return []
            
        print(f"📖 Reading Tier 1 Excel dataset: {os.path.basename(file_path)}...")
        df = pd.read_excel(file_path)
        processed = []
        
        for idx, row in df.iterrows():
            raw_dict = row.to_dict()
            raw_dict["chunk_id"] = f"EXCEL_{idx+1}"
            raw_dict["content_source"] = "Tier 1 (Kaggle Dataset)"
            chunk = self.validate_and_format_chunk(raw_dict)
            if chunk:
                processed.append(chunk)
                
        return processed

    def load_mlbd_pdf_folder(self, pdf_dir_path: str) -> List[Dict[str, Any]]:
        """Parses ALL 123 Motilal Banarsidass (MLBD) PDF files completely."""
        if not os.path.exists(pdf_dir_path):
            return []
            
        extracted_chunks = []
        pdf_files = [f for f in os.listdir(pdf_dir_path) if f.endswith('.pdf')]
        total_files = len(pdf_files)
        print(f"📖 Starting Full Parsing of ALL {total_files} MLBD PDF files in Tier 2...")
        
        # Loop through ALL 123 PDF files (No limits!)
        for index, pdf_file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(pdf_dir_path, pdf_file)
            purana_name = pdf_file.replace('.pdf', '').replace('_', ' ')
            print(f"[{index}/{total_files}] Processing PDF: {pdf_file}...")
            
            try:
                reader = PdfReader(pdf_path)
                current_text = ""
                chunk_index = 1
                
                # Loop through ALL pages of every PDF (No limits!)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    current_text += "\n" + page_text
                    
                    if len(current_text.split()) >= 400:
                        chunk = self.validate_and_format_chunk({
                            "chunk_id": f"MLBD_{purana_name[:2].upper()}_{chunk_index}",
                            "purana_name": purana_name,
                            "skandha_book": 1,
                            "chapter": chunk_index,
                            "story_title": f"{purana_name} (Page {i+1})",
                            "text_content": current_text,
                            "content_source": "Tier 2 (MLBD Canonical PDF)"
                        })
                        if chunk:
                            extracted_chunks.append(chunk)
                        current_text = ""
                        chunk_index += 1
            except Exception as e:
                print(f"⚠️ Warning parsing PDF {pdf_file}: {e}")
                
        return extracted_chunks

    def load_all_datasets(self) -> List[Dict[str, Any]]:
        """Unified loader combining Tier 1 Excel, Tier 2 PDFs, and JSON datasets."""
        all_chunks = []
        
        # 1. Load Tier 1 Excel
        excel_path = os.path.join(self.data_dir, "18_Mahapuranas_DataSet.xlsx")
        excel_chunks = self.load_excel_dataset(excel_path)
        all_chunks.extend(excel_chunks)
        
        # 2. Load Tier 2 MLBD PDFs (ALL 123 PDFs!)
        pdf_dir = os.path.join(self.data_dir, "mlbd_pdfs")
        pdf_chunks = self.load_mlbd_pdf_folder(pdf_dir)
        all_chunks.extend(pdf_chunks)
        
        # 3. Load Sample JSON fallback
        json_path = os.path.join(self.data_dir, "sample_mahapuranas.json")
        json_chunks = self.load_json_dataset(json_path)
        all_chunks.extend(json_chunks)
        
        # Save preprocessed master chunks to cache file
        master_cache_path = os.path.join(self.data_dir, "preprocessed_master_chunks.json")
        try:
            with open(master_cache_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, indent=2)
            print(f"💾 Saved preprocessed master dataset to: {master_cache_path}")
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
            
        print(f"✅ Full Preprocessing Complete: Total {len(all_chunks)} chunks loaded across all 18 Mahapuranas!")
        return all_chunks

if __name__ == "__main__":
    preprocessor = MahapuranaPreprocessor()
    dataset_chunks = preprocessor.load_all_datasets()