from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from component.utils import all_chatbot_utils
import numpy as np

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Initialize conversation history
system_prompt = "You are a kind and helpful assistant. Your responses must be short and precise. Only use the information provided in the system context to answer the user's question. Do not rely on or generate responses based on any prior or pre-trained knowledge. If the answer cannot be found in the given context, respond with 'I don't know based on the provided information.' Do not make assumptions or hallucinate facts."
messages = [{"role": "system", "content":system_prompt}]


if __name__ == "__main__":
    obj1 = all_chatbot_utils()
    chunks1 = obj1.get_chunks()
    index = obj1.store_chunks_in_faiss(chunks1)

    # Start chat loop
    print("🤖 I'm your virtual assistant here to answer your queries! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        #Retrieve the similar chunks/context from Faiss vector database
        retrieved_context = obj1.fetch_similar_chunks(user_input,chunks1,index)
        # Append the retrieved context
        messages.append({"role": "system", "content":retrieved_context})
        # Append user input to context
        messages.append({"role": "user", "content":user_input})

        # prepping the prompt in Qwen's expected chat format
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False 
        )
        # tokenize the structured chat string into model-ready input.
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        # taking newly generated token IDs
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")


        # Add assistant response to messages
        messages.append({"role":"assistant", "content":content})
        print(messages)
        print("Assistant:", content)


        
