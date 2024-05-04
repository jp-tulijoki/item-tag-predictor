import json
import matplotlib.pyplot as plt

# these files were made by copy-pasting the bert_masked_lm output json to file
movie_path = "movie_mlm_losses.txt"
book_path = "book_mlm_losses.txt"

movie_losses = []
book_losses = []

with open (movie_path, "r") as file:
    for line in file:
        line = line.replace("'","\"")
        json_dict = json.loads(line)
        movie_losses.append(json_dict["train_loss"])

with open (book_path, "r") as file:
    for line in file:
        line = line.replace("'","\"")
        json_dict = json.loads(line)
        book_losses.append(json_dict["train_loss"])        

plt.figure(figsize=(10, 6))
plt.plot(movie_losses, color="skyblue", label="Movies")
plt.plot(book_losses, color="red", label="Books")
plt.title("Masked language model losses")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()       