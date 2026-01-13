import time
from google import genai
from google.api_core import exceptions
# 1. Initialize the Client
# It will automatically look for the GEMINI_API_KEY environment variable, 
# or you can pass it explicitly: api_key="YOUR_KEY_HERE"
#client = genai.Client()

api_key="AIzaSyA4u4z-LL-bMYu2bHXWxUQC8f24UabwuVE"
client = genai.Client(api_key=api_key)

def test_all_models():
    print("--- Fetching Available Models ---")
    
    # 2. Get the list of all models
    try:
        models = list(client.models.list())
        model_names = [m.name for m in models]
        print(f"Found {len(model_names)} total models.\n")
    except Exception as e:
        print(f"Error fetching model list: {e}")
        return

    # 3. Filter for models that support text generation (to avoid testing embeddings/etc.)
    # The list format is usually 'models/gemini-1.5-flash'
    testable_models = [m for m in models if "generateContent" in m.supported_actions]

    print(f"Testing {len(testable_models)} models that support generation...\n")
    print(f"{'Model Name':<35} | {'Status':<15} | {'Response Snippet'}")
    print("-" * 80)

    for model_meta in testable_models:
        model_id = model_meta.name
        
        try:
            # Short test prompt
            response = client.models.generate_content(
                model=model_id, 
                contents="Say 'System Online'"
            )
            
            status = "✅ Success"
            snippet = response.text.strip().replace("\n", " ")[:30]
            print(f"{model_id:<35} | {status:<15} | {snippet}...")

        except exceptions.ResourceExhausted:
            print(f"{model_id:<35} | ❌ Quota Limit | (Wait before retrying)")
            # Pause to avoid spamming the API further if you're hit with 429s
            time.sleep(2) 
        except Exception as e:
            # Catching specific model errors (some models might be restricted)
            print(f"{model_id:<35} | ❌ Error       | {str(e)[:40]}...")

if __name__ == "__main__":
    test_all_models()