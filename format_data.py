import os
import json
import numpy as np
import pickle  # Biblioteca pentru serializare

# Functie pentru a obtine numarul de actiuni de tip "ROLL" dintr-un fisier JSON
def get_roll_counts(file_path):
    try:
        # Deschide fisierul si incarca datele JSON
        with open(file_path, "r") as file:
            data = json.load(file)

        # Obtine actiunile din fisier
        actions = data.get("actions", [])
        roll_count = 0

        # Numara actiunile de tip "ROLL"
        for action in actions:
            if len(action) > 1 and action[1] == "ROLL":
                roll_count += 1

        # Rotunjeste in sus jumatatea numarului total de actiuni "ROLL"
        rounded_rolls = int(np.ceil(roll_count / 2) + 1)
        return rounded_rolls
    except Exception as e:
        # Intoarce un mesaj de eroare daca fisierul nu poate fi procesat
        print(f"Eroare la procesare {file_path}: {e}")
        return 0

def process_folder(folder_path):
    rounds_array = []
    for file_name in os.listdir(folder_path):
        # Proceseaza doar fisierele cu extensia ".json"
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            roll_count = get_roll_counts(file_path)
            if roll_count < 400:
                rounds_array.append(roll_count)
    return rounds_array

def save_results_to_pickle(folder_key, data, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"sim_{folder_key}.pkl")
    try:
        with open(output_file, "wb") as file:
            pickle.dump(data, file)
        print(f"Salvat {output_file}")
    except Exception as e:
        print(f"Eroare {folder_key} la: {e}")

if __name__ == "__main__":
    # Definirea cailor pentru folderele de procesat
    folders = {
        "R_R": "data/R_R_data",
        "W_W": "data/W_W_data",
        "VP_VP": "data/VP_VP_data",
        "R_W": "data/R_W_data",
        "R_VP": "data/R_VP_data",
        "W_VP": "data/W_VP_data",
        "AP_R_W_VP": "data/AP_R_W_VP_data",
    }

    # Proceseaza fiecare folder definit
    for key, folder_path in folders.items():
        if os.path.exists(folder_path):
            # Proceseaza fisierele din folder
            processed_data = process_folder(folder_path)
            # Salveaza rezultatele intr-un fisier .pkl
            save_results_to_pickle(key, processed_data)
        else:
            print(f"Folder {folder_path} nu exista")
