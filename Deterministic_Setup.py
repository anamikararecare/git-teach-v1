import os
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, disk_offload
import torch
from huggingface_hub import login
login(new_session=False)

# Replace your openai api key here
# OPENAI_API_KEY = "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"
# llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

def detect_tech_stack(repo_path):
    tech_stack = {
        "languages": set(),
        "frameworks": set(),
        "build_tools": set(),
        "databases": set(),
    }
    
    filetypes = {'.cpp': 'C/C++',
                 '.cs': 'C#',
                 '.css': 'CSS',
                 '.dart': 'Dart',
                 'Dockerfile': 'Dockerfile',
                 '.fs': 'F#',
                 '.go': 'Go',
                 '.html': 'HTML',
                 '.java': 'Java',
                 '.js': 'JavaScript',
                 '.json': 'JSON',
                 '.jl': 'Julia',
                 '.less': 'Less',
                 '.md': 'Markdown',
                 '.php': 'PHP',
                 '.ps1': 'PowerShell',
                 '.py': 'Python',
                 '.r': 'R',
                 '.rb': 'Ruby',
                 '.rs': 'Rust',
                 '.scss': 'SCSS',
                 '.swift': 'Swift',
                 '.ts': 'TypeScript',
                 '.sql': 'T-SQL'
                 }
    
    frameworks_to_files = {"Docker": "docker-compose.yml",
                           "Make": "Makefile",
                           "Vagrant": "Vagrantfile",
                           "Kubernetes": ["k8s.yaml", "deployment.yaml"],
                           "Terraform": "main.tf",
                           "CircleCI": ".circleci/config.yml",
                           "GitHub Actions": ".github/workflows/",
                           "Jenkins": "Jenkinsfile"
                           }
    
    for tool, filenames in frameworks_to_files.items():
        if isinstance(filenames, list):
            if any(file == f for f in filenames for file in os.listdir(repo_path)):
                tech_stack["build_tools"].add(tool)

    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            _, ext = os.path.splitext(file)
            if ext in filetypes:
                tech_stack["languages"].add(filetypes[ext])
            elif file == "requirements.txt":
                tech_stack["languages"].add("Python")
                _parse_requirements(file_path, tech_stack)
            elif file == "package.json":
                tech_stack["languages"].add("JavaScript")
                _parse_package_json(file_path, tech_stack)
            elif file == "Cargo.toml":
                tech_stack["languages"].add("Rust")
            elif file == "go.mod":
                tech_stack["languages"].add("Go")
            elif file.endswith(".csproj"):
                tech_stack["languages"].add("C#")
            
            # === Framework Detection (File-Based) ===
            if "docker-compose.yml" in file:
                tech_stack["build_tools"].add("Docker")
            if "Makefile" in file:
                tech_stack["build_tools"].add("Make")
    
    _detect_frameworks_from_imports(repo_path, tech_stack)
    
    return tech_stack

def _parse_requirements(file_path, tech_stack):
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                # Extract package name (ignore versions)
                pkg = line.split("==")[0].split(">")[0].split("<")[0]
                
                # Map packages to frameworks
                if pkg in ["django", "flask", "fastapi"]:
                    tech_stack["frameworks"].add(pkg.capitalize())
                elif pkg in ["torch", "tensorflow"]:
                    tech_stack["frameworks"].add("PyTorch" if pkg == "torch" else "TensorFlow")
                elif pkg in ["psycopg2", "sqlalchemy"]:
                    tech_stack["databases"].add("PostgreSQL" if pkg == "psycopg2" else "SQLAlchemy")
    except FileNotFoundError:
        pass

def _parse_package_json(file_path, tech_stack):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            deps = data.get("dependencies", {}) | data.get("devDependencies", {})
            
            for pkg in deps:
                if pkg in ["react", "vue", "angular"]:
                    tech_stack["frameworks"].add(pkg.capitalize())
                elif pkg in ["express", "koa"]:
                    tech_stack["frameworks"].add("Express.js" if pkg == "express" else "Koa")
                elif pkg in ["mongoose", "sequelize"]:
                    tech_stack["databases"].add("MongoDB" if pkg == "mongoose" else "SQL")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

def _detect_frameworks_from_imports(repo_path, tech_stack):
    # Only run if language is Python/JS (for brevity)
    if "Python" not in tech_stack["languages"] and "JavaScript" not in tech_stack["languages"]:
        return
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                _scan_python_imports(os.path.join(root, file), tech_stack)
            elif file.endswith(".js") or file.endswith(".jsx"):
                _scan_js_imports(os.path.join(root, file), tech_stack)

def _scan_python_imports(file_path, tech_stack):
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith(("import ", "from ")):
                    if "django" in line:
                        tech_stack["frameworks"].add("Django")
                    elif "flask" in line:
                        tech_stack["frameworks"].add("Flask")
                    elif "torch" in line:
                        tech_stack["frameworks"].add("PyTorch")
                    elif "fastapi" in line:
                        tech_stack["frameworks"].add("FastAPI")
                    elif "pyramid" in line:
                        tech_stack["frameworks"].add("Pyramid")
    except FileNotFoundError:
        pass

def _scan_js_imports(file_path, tech_stack):
    try:
        with open(file_path, "r") as f:
            for line in f:
                if "require(" in line or "from " in line or "import " in line:
                    if "react" in line:
                        tech_stack["frameworks"].add("React")
                    elif "vue" in line:
                        tech_stack["frameworks"].add("Vue.js")
                    elif "express" in line:
                        tech_stack["frameworks"].add("Express.js")
                    elif "angular" in line:
                        tech_stack["frameworks"].add("Angular")
                    elif "next" in line:
                        tech_stack["frameworks"].add("Next.js")
                    elif "nuxt" in line:
                        tech_stack["frameworks"].add("Nuxt.js")
    except FileNotFoundError:
        pass

def generate_setup_guide(tech_stack):
    question = f"""
The project uses the following technologies:
- Languages: {', '.join(tech_stack["languages"])}
- Frameworks: {', '.join(tech_stack["frameworks"])}
- Databases: {', '.join(tech_stack["databases"])}

Generate a beginner-friendly guide to set up this project in VSCode:
1. What type of project is this (e.g., web app, ML model)?
2. List ONLY ESSENTIAL tools to install (e.g. languages).
3. Explain how to configure the environment.
4. Why is this tech stack used? What are alternatives generally used for this project type?
"""
    model_id = "tiiuae/falcon-7b-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model = disk_offload(model, offload_dir="./pycache")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pl = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    response = pl(question)

    print(response)

boilerplate_files = { "README.md", 
                     "LICENSE", 
                     ".gitignore", 
                     "requirements.txt", 
                     "package.json", 
                     "Dockerfile", 
                     "Makefile", 
                     "github", 
                     "node_modules", 
                     "venv",
                     "setup.py",
                     "package-lock.json",
                     "yarn.lock",}

def filter_boilerplate_files(repo_path):
    result = {"boilerplate_files": [], "main_code": []}
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file in boilerplate_files:
                result["boilerplate_files"].append(os.path.join(root, file))
            else:
                result["main_code"].append(os.path.join(root, file))
    
    return result