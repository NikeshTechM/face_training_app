import os
print("Loaded API Key:", "YES" if os.getenv("OPENAI_API_KEY") else "NO")
import openai

# Create OpenAI client with API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_all_code_files(base_path):
    code_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.py', '.js', '.cpp', '.java')):
                full_path = os.path.join(root, file)
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    code_files.append((full_path, content))
    return code_files

def ask_llm_to_analyze(file_path, content):
    prompt = f"""
You are a static code analyzer. Analyze the following code for memory leak issues.

File: {file_path}
Code:
{content}

If there are memory leak issues, fix them and return the entire modified code file. 
If no issues, return the original code.
Respond only with the full code.
"""
    response = client.chat.completions.create(
        model="gpt-4o",  # or gpt-4
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def write_back_if_modified(path, new_content):
    with open(path, 'r', encoding='utf-8') as f:
        original = f.read()
    if original != new_content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {path}")
    else:
        print(f"No changes: {path}")

def main():
    code_files = read_all_code_files('.')
    for file_path, content in code_files:
        print(f"Analyzing: {file_path}")
        result = ask_llm_to_analyze(file_path, content)
        # Remove ``` blocks if present
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith(("python", "cpp", "java", "js")):
                result = "\n".join(result.split("\n")[1:])
        write_back_if_modified(file_path, result)

if __name__ == "__main__":
    main()
