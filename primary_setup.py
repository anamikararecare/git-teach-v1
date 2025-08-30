import os
import git
import typer

from deterministic_setup import detect_tech_stack, generate_setup_guide, filter_boilerplate_files
from chunking import *

from google import genai
from google.genai.types import HttpOptions

app = typer.Typer()

@app.command()
def fetch_repo(repo_url: str, skill_level: str = typer.Option("beginner", help="Skill level: beginner/intermediate/advanced")):
    repo_name = repo_url.rstrip("/").split("/")[-1]
    clone_dir = f"./cloned_repos/{repo_name}"


    if os.path.exists(clone_dir):
        typer.echo(f"Repository {repo_name} already exists in {clone_dir}. Skipping cloning...")
    else:
        typer.echo(f"Cloning repository {repo_url} into {clone_dir}...")
        git.Repo.clone_from(repo_url, clone_dir)

        typer.echo(f"Skill level set to {skill_level}.")
        typer.echo(f"Repository {repo_name} cloned successfully into {clone_dir}.")
    
    typer.echo("\nAnalyzing tech stack...")
    tech_stack = detect_tech_stack(clone_dir)
    typer.echo(f"Detected Tech Stack: {tech_stack}")

    typer.echo("\n Filtering boilerplate code...")
    filtered_code = filter_boilerplate_files(clone_dir)
    # typer.echo(f"Found {len(filtered_code)} relevant files.")
    typer.echo(f"The relevant files are: {filtered_code}")

    typer.echo("\nGenerating setup guide...")
    setup_guide = generate_setup_guide(tech_stack)
    typer.echo(f"\n=== Setup Guide ===\n{setup_guide}")

    files = retrieve_file_list(filtered_code)
    chapter_guide = index_repository(files)
    typer.echo(f"Summary: {chapter_guide['summary']}")

    question = f"""
Given this chapter list (JSON), propose a high-level chapter itinerary with 1-line summaries per chapter. 
You do not need to conform to this chapter list - you may combine functions where logical.
Do not invent code beyond what’s referenced. Return a numbered list. 
{chapter_guide}
"""
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
    )

    typer.echo(response.text)
    next_question = typer.prompt("Ready to start?")
    while next_question != "exit":
        next_question = typer.prompt("Ready?")
        new_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(next_question + f"{chapter_guide}"),
    )
        typer.echo(new_response.text)

if __name__ == "__main__":
    app()

