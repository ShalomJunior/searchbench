import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class RAGGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", quantize: bool = False):
        """
        Initialize the RAG Generator using a HuggingFace text-generation model.
        Default is a very small model (0.5B) suitable for local CPU testing.
        When running on Kaggle with GPUs, you can pass a larger model like:
        'meta-llama/Meta-Llama-3-8B-Instruct' or 'Qwen/Qwen2.5-7B-Instruct'.
        """
        self.model_name = model_name
        print(f"Loading LLM: {model_name}...")
        
        # Determine device (MPS for Mac, CUDA for Nvidia, CPU fallback)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            
        print(f"Using device: {device}")
        print(f"Quantization enabled: {quantize}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model_kwargs = {
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
            "low_cpu_mem_usage": True
        }
        
        if quantize and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["device_map"] = "auto"
            except ImportError:
                print("Warning: bitsandbytes not installed. Falling back to unquantized loading.")
        elif device == "cuda":
            model_kwargs["device_map"] = "auto"
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not (quantize and device == "cuda") and "device_map" not in model_kwargs:
            self.model.to(device) # type: ignore
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=False, # We want factual, deterministic answers
            temperature=None,
            top_p=None
        )
        
    def build_prompt(self, query: str, documents: list[dict]) -> str:
        """
        Constructs the RAG prompt with strict instructions for citations.
        """
        system_prompt = (
            "You are an expert scientific research assistant. "
            "Your task is to answer the user's question based STRICTLY on the provided documents.\n"
            "If the provided documents do not contain the answer, explicitly state that you do not know.\n"
            "When you use information from a document, you MUST cite it using the format [doc_id] at the end of the sentence.\n\n"
        )
        
        docs_text = "Documents:\n"
        for doc in documents:
            docs_text += f"[{doc['doc_id']}] {doc['text']}\n\n"
            
        user_prompt = f"Question: {query}\nAnswer:"
        
        # Check if the model has a chat template
        if self.tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": docs_text + user_prompt}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
        else:
            # Fallback for models without chat templates
            return f"{system_prompt}{docs_text}{user_prompt}"

    def generate(self, query: str, documents: list[dict]) -> str:
        """
        Generates an answer with citations based on the query and retrieved documents.
        """
        prompt = self.build_prompt(query, documents)
        
        # Generate text
        outputs = self.pipe(prompt, return_full_text=False)
        answer = outputs[0]["generated_text"].strip() # type: ignore
        
        return answer
