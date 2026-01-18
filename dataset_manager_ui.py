#!/usr/bin/env python3
"""
Dataset Manager - Web UI for managing training data

Features:
- View all samples in the dataset
- Add new samples
- Edit existing samples
- Delete samples
- Search and filter samples
- Save changes to dataset
"""

import json
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime


class DatasetManager:
    """Manages loading, editing, and saving training datasets."""

    def __init__(self, dataset_path: str = "data/processed/train_data.json"):
        self.dataset_path = Path(dataset_path)
        self.backup_dir = Path("data/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.data: List[Dict] = []
        self.load_dataset()

    def load_dataset(self):
        """Load dataset from JSON file."""
        if self.dataset_path.exists():
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ Loaded {len(self.data)} samples from {self.dataset_path}")
        else:
            print(f"✗ Dataset not found: {self.dataset_path}")
            self.data = []

    def save_dataset(self, backup: bool = True):
        """Save dataset to JSON file with optional backup."""
        if backup and self.dataset_path.exists():
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{self.dataset_path.stem}_backup_{timestamp}.json"
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                backup_data = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(backup_data)
            print(f"✓ Backup saved: {backup_path}")

        # Save current data
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"✓ Dataset saved: {self.dataset_path} ({len(self.data)} samples)")

    def get_sample(self, index: int) -> Optional[Dict]:
        """Get a sample by index."""
        if 0 <= index < len(self.data):
            return self.data[index]
        return None

    def add_sample(self, question: str, answer: str, source: str,
                   url: str = "", title: str = "") -> str:
        """Add a new sample to the dataset."""
        if not question or not answer:
            return "❌ Question and answer are required"

        new_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un experto en regulación bancaria española. Tu tarea es responder preguntas sobre regulación bancaria, especialmente sobre parámetros de riesgo de crédito.\n\nIMPORTANTE:\n- Siempre cita la fuente de tu información\n- Si no estás seguro o no tienes la información, responde \"No tengo información suficiente para responder con certeza\" o \"Esta información no está disponible en los documentos proporcionados\"\n- Nunca inventes información\n- Sé preciso y específico en tus respuestas"
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ],
            "metadata": {
                "source": source or "Manual Entry",
                "url": url or "",
                "title": title or "Manual Entry"
            }
        }

        self.data.append(new_sample)
        return f"✅ Added new sample (Total: {len(self.data)})"

    def update_sample(self, index: int, question: str, answer: str,
                     source: str, url: str = "", title: str = "") -> str:
        """Update an existing sample."""
        if not 0 <= index < len(self.data):
            return f"❌ Invalid index: {index}"

        if not question or not answer:
            return "❌ Question and answer are required"

        self.data[index]["messages"][1]["content"] = question
        self.data[index]["messages"][2]["content"] = answer
        self.data[index]["metadata"]["source"] = source or "Manual Entry"
        self.data[index]["metadata"]["url"] = url or ""
        self.data[index]["metadata"]["title"] = title or "Manual Entry"

        return f"✅ Updated sample #{index}"

    def delete_sample(self, index: int) -> str:
        """Delete a sample by index."""
        if not 0 <= index < len(self.data):
            return f"❌ Invalid index: {index}"

        deleted = self.data.pop(index)
        question = deleted["messages"][1]["content"][:100]
        return f"✅ Deleted sample #{index}: {question}... (Remaining: {len(self.data)})"

    def search_samples(self, query: str, search_in: str = "all") -> pd.DataFrame:
        """Search samples by query."""
        results = []
        query_lower = query.lower() if query else ""

        for i, sample in enumerate(self.data):
            question = sample["messages"][1]["content"]
            answer = sample["messages"][2]["content"]
            source = sample["metadata"].get("source", "")

            # Determine if sample matches query
            match = False
            if not query:
                match = True
            elif search_in == "question" and query_lower in question.lower():
                match = True
            elif search_in == "answer" and query_lower in answer.lower():
                match = True
            elif search_in == "source" and query_lower in source.lower():
                match = True
            elif search_in == "all" and (
                query_lower in question.lower() or
                query_lower in answer.lower() or
                query_lower in source.lower()
            ):
                match = True

            if match:
                results.append({
                    "Index": i,
                    "Question": question[:150] + "..." if len(question) > 150 else question,
                    "Answer": answer[:150] + "..." if len(answer) > 150 else answer,
                    "Source": source
                })

        return pd.DataFrame(results)

    def get_stats(self) -> str:
        """Get dataset statistics."""
        if not self.data:
            return "No data loaded"

        sources = {}
        total_q_chars = 0
        total_a_chars = 0

        for sample in self.data:
            source = sample["metadata"].get("source", "Unknown")
            sources[source] = sources.get(source, 0) + 1
            total_q_chars += len(sample["messages"][1]["content"])
            total_a_chars += len(sample["messages"][2]["content"])

        avg_q = total_q_chars // len(self.data)
        avg_a = total_a_chars // len(self.data)

        stats = f"""📊 Dataset Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples: {len(self.data)}
Average Question Length: {avg_q} characters
Average Answer Length: {avg_a} characters

Sources:
"""
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.data)) * 100
            stats += f"  • {source}: {count} ({percentage:.1f}%)\n"

        return stats


# Initialize manager
manager = DatasetManager()


# Gradio Interface Functions

def view_sample(index: int) -> tuple:
    """View a sample by index."""
    sample = manager.get_sample(index)
    if not sample:
        return "Invalid index", "", "", "", ""

    question = sample["messages"][1]["content"]
    answer = sample["messages"][2]["content"]
    source = sample["metadata"].get("source", "")
    url = sample["metadata"].get("url", "")
    title = sample["metadata"].get("title", "")

    return question, answer, source, url, title


def add_sample_handler(question: str, answer: str, source: str, url: str, title: str) -> str:
    """Handle adding a new sample."""
    result = manager.add_sample(question, answer, source, url, title)
    manager.save_dataset(backup=True)
    return result


def update_sample_handler(index: int, question: str, answer: str,
                         source: str, url: str, title: str) -> str:
    """Handle updating a sample."""
    result = manager.update_sample(index, question, answer, source, url, title)
    manager.save_dataset(backup=True)
    return result


def delete_sample_handler(index: int) -> str:
    """Handle deleting a sample."""
    result = manager.delete_sample(index)
    manager.save_dataset(backup=True)
    return result


def search_handler(query: str, search_in: str) -> pd.DataFrame:
    """Handle search."""
    return manager.search_samples(query, search_in)


def refresh_stats() -> str:
    """Refresh statistics."""
    manager.load_dataset()
    return manager.get_stats()


def load_dataset_handler(dataset_path: str) -> str:
    """Load a different dataset."""
    manager.dataset_path = Path(dataset_path)
    manager.load_dataset()
    return f"✓ Loaded {len(manager.data)} samples from {dataset_path}"


# Build Gradio Interface

with gr.Blocks(title="RegLLM Dataset Manager", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 RegLLM Dataset Manager")
    gr.Markdown("Manage your training dataset: view, add, edit, and delete Q&A samples")

    with gr.Tab("📋 Browse & Search"):
        gr.Markdown("### Search and Browse Dataset")

        with gr.Row():
            search_query = gr.Textbox(label="Search Query", placeholder="Enter search term...")
            search_in = gr.Radio(
                choices=["all", "question", "answer", "source"],
                value="all",
                label="Search In"
            )
            search_btn = gr.Button("🔍 Search", variant="primary")

        search_results = gr.Dataframe(
            label="Search Results",
            interactive=False,
            wrap=True
        )

        search_btn.click(
            fn=search_handler,
            inputs=[search_query, search_in],
            outputs=search_results
        )

        # Auto-search on load
        demo.load(fn=lambda: manager.search_samples("", "all"), outputs=search_results)

    with gr.Tab("👁️ View Sample"):
        gr.Markdown("### View Sample Details")

        view_index = gr.Number(label="Sample Index", value=0, precision=0)
        view_btn = gr.Button("👁️ Load Sample", variant="primary")

        with gr.Row():
            with gr.Column():
                view_question = gr.Textbox(label="Question", lines=5, interactive=False)
                view_answer = gr.Textbox(label="Answer", lines=10, interactive=False)
            with gr.Column():
                view_source = gr.Textbox(label="Source", interactive=False)
                view_url = gr.Textbox(label="URL", interactive=False)
                view_title = gr.Textbox(label="Title", interactive=False)

        view_btn.click(
            fn=view_sample,
            inputs=view_index,
            outputs=[view_question, view_answer, view_source, view_url, view_title]
        )

    with gr.Tab("➕ Add Sample"):
        gr.Markdown("### Add New Sample")

        add_question = gr.Textbox(label="Question", lines=3, placeholder="Enter the question...")
        add_answer = gr.Textbox(label="Answer", lines=8, placeholder="Enter the answer...")

        with gr.Row():
            add_source = gr.Textbox(label="Source", placeholder="e.g., EBA, Bank of Spain")
            add_url = gr.Textbox(label="URL (optional)", placeholder="https://...")

        add_title = gr.Textbox(label="Title (optional)", placeholder="Document title")

        add_btn = gr.Button("➕ Add Sample", variant="primary")
        add_status = gr.Textbox(label="Status", interactive=False)

        add_btn.click(
            fn=add_sample_handler,
            inputs=[add_question, add_answer, add_source, add_url, add_title],
            outputs=add_status
        )

    with gr.Tab("✏️ Edit Sample"):
        gr.Markdown("### Edit Existing Sample")

        edit_index = gr.Number(label="Sample Index to Edit", value=0, precision=0)
        load_edit_btn = gr.Button("📥 Load Sample for Editing")

        edit_question = gr.Textbox(label="Question", lines=3)
        edit_answer = gr.Textbox(label="Answer", lines=8)

        with gr.Row():
            edit_source = gr.Textbox(label="Source")
            edit_url = gr.Textbox(label="URL")

        edit_title = gr.Textbox(label="Title")

        update_btn = gr.Button("💾 Update Sample", variant="primary")
        edit_status = gr.Textbox(label="Status", interactive=False)

        load_edit_btn.click(
            fn=view_sample,
            inputs=edit_index,
            outputs=[edit_question, edit_answer, edit_source, edit_url, edit_title]
        )

        update_btn.click(
            fn=update_sample_handler,
            inputs=[edit_index, edit_question, edit_answer, edit_source, edit_url, edit_title],
            outputs=edit_status
        )

    with gr.Tab("🗑️ Delete Sample"):
        gr.Markdown("### Delete Sample")
        gr.Markdown("⚠️ **Warning**: Deletion is permanent (backup will be created)")

        delete_index = gr.Number(label="Sample Index to Delete", value=0, precision=0)
        preview_delete_btn = gr.Button("👁️ Preview Sample")

        with gr.Row():
            preview_question = gr.Textbox(label="Question Preview", lines=3, interactive=False)
            preview_answer = gr.Textbox(label="Answer Preview", lines=5, interactive=False)

        delete_btn = gr.Button("🗑️ Delete Sample", variant="stop")
        delete_status = gr.Textbox(label="Status", interactive=False)

        preview_delete_btn.click(
            fn=view_sample,
            inputs=delete_index,
            outputs=[preview_question, preview_answer, gr.Textbox(visible=False),
                    gr.Textbox(visible=False), gr.Textbox(visible=False)]
        )

        delete_btn.click(
            fn=delete_sample_handler,
            inputs=delete_index,
            outputs=delete_status
        )

    with gr.Tab("📊 Statistics"):
        gr.Markdown("### Dataset Statistics")

        refresh_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
        stats_output = gr.Textbox(label="Statistics", lines=20, interactive=False)

        refresh_btn.click(fn=refresh_stats, outputs=stats_output)
        demo.load(fn=refresh_stats, outputs=stats_output)

    with gr.Tab("⚙️ Settings"):
        gr.Markdown("### Dataset Settings")

        dataset_path_input = gr.Textbox(
            label="Dataset Path",
            value=str(manager.dataset_path),
            placeholder="data/processed/train_data.json"
        )
        load_dataset_btn = gr.Button("📂 Load Dataset", variant="primary")
        load_status = gr.Textbox(label="Status", interactive=False)

        load_dataset_btn.click(
            fn=load_dataset_handler,
            inputs=dataset_path_input,
            outputs=load_status
        )

        gr.Markdown("### Backup Information")
        gr.Markdown(f"Backups are automatically created in: `{manager.backup_dir}`")
        gr.Markdown("Each save operation creates a timestamped backup before overwriting.")


if __name__ == "__main__":
    print("=" * 70)
    print("RegLLM Dataset Manager - Web UI")
    print("=" * 70)
    print()
    print(f"Dataset: {manager.dataset_path}")
    print(f"Samples: {len(manager.data)}")
    print(f"Backups: {manager.backup_dir}")
    print()
    print("Starting web interface...")
    print()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
