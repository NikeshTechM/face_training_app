import openai
import os
import json

# Set OpenAI key via GitHub Secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_all_code_files(base_path):
    code_files = []
    for root, dirs, files in os.walk(base_path):
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

Respond with the fixed code if any issues are found.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # or gpt-4, gpt-4-1106-preview
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

def main():
    code_files = read_all_code_files('.')
    for file_path, content in code_files:
        print(f"Analyzing: {file_path}")
        result = ask_llm_to_analyze(file_path, content)
        if result.startswith("```"):  # strip code block markers
            result = result.split("```")[1]
            if result.startswith(("python", "cpp", "java", "js")):
                result = "\n".join(result.split("\n")[1:])
        write_back_if_modified(file_path, result)

if __name__ == "__main__":
    main()
