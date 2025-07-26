from tokenization import chunk_repo, generate_embeddings, save_embeddings

def process_repo(repo_path:str, output_dir:str):
    chunks = chunk_repo(repo_path)
    print(chunks)

    embeddings = generate_embeddings(chunks)

    save_embeddings(embeddings, f"{output_dir}/embeddings.npy")
    print(f"Generated {len(chunks)} chunks with embeddings of shape {embeddings.shape}")

process_repo("./cloned_repos/Portfolio-Website", "./output")