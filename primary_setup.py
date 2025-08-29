import os
import git
import typer

from deterministic_setup import detect_tech_stack, generate_setup_guide, filter_boilerplate_files

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
    typer.echo(f"Found {len(filtered_code["main_code"])} relevant files.")
    typer.echo(f"The relevant files are: {filtered_code["main_code"]}")


    typer.echo("\nGenerating setup guide...")
    setup_guide = generate_setup_guide(tech_stack)
    typer.echo(f"\n=== Setup Guide ===\n{setup_guide}")

    

if __name__ == "__main__":
    app()

