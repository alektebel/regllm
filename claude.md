




Finetuning project for spanish banking regulation compliance using a custom fine tuned ollama model.

Startig point is the ollama 3b.

We need to scrape a dataset of spanish banking regulation documents, create a pipeline to clean and preprocess the data, and then fine tune the model using the cleaned dataset.
The objective is to create a lightweight model that can run locally under 4GM of RAM and provide accurate responses to questions related to spanish banking regulations.

Steps to follow:

- Create a web scraper to collect spanish banking regulation documents from official sources. Basel, ECB, Bank of Spain, BOE, CNMV. Everything related to credit risk parameter regulation on spanish banks.
- Store the scraped documents in a structured format (e.g. JSON, CSV).
- Preprocess the data to remove any irrelevant information, such as HTML tags, advertisements, and other non-regulatory content.
- Split the data into training and validation sets.
- Fine tune the ollama 3b model using the training set, ensuring that the model can run efficiently within the 4GB RAM constraint, adjusting all the parameters of the network.
-   First, try to overfit a small subset of the data to ensure the fine-tuning process is working correctly.
    The case use of the model will be to receive information related to specifics of spanish banking regulation compliance (for instance, in bank X we are calculating the PD of a retail portfolio, what regulation applies to this case?)
    model shoulb cite the source of the information provided always and without making things up ever. In case the model is not sure of the answer, or that it need information not present in the provided prompt, it should respond with "I don't know" or "the information is not available in the provided documents.".

Once the small subset is overfitted, proceed to fine tune the model using the full training set.
- Evaluate the model's performance using the validation set, focusing on accuracy and relevance of the responses
- Plot progress of training and validation loss over epochs to monitor for progress and overfitting.
- Make sure to use the GPU for training and speed up the process.
- Make a simple interface to interact with the fine-tuned model, allowing users to input questions and receive responses related to spanish banking regulations.


