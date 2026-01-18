#!/usr/bin/env python3
"""
Dataset Manager CLI - Command-line interface for dataset management

This CLI allows programmatic access to the dataset for LLMs, agents, and scripts.
Supports JSON input/output for easy integration with other tools.

Usage:
    # List all samples
    python dataset_manager_cli.py list

    # Get specific sample
    python dataset_manager_cli.py get <index>

    # Search samples
    python dataset_manager_cli.py search "query" [--field all|question|answer|source]

    # Add sample (JSON input)
    python dataset_manager_cli.py add --question "..." --answer "..." --source "..."

    # Update sample
    python dataset_manager_cli.py update <index> --question "..." --answer "..."

    # Delete sample
    python dataset_manager_cli.py delete <index>

    # Get statistics
    python dataset_manager_cli.py stats

    # Export to JSON
    python dataset_manager_cli.py export --output output.json

    # Import from JSON
    python dataset_manager_cli.py import --input input.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional


class DatasetManagerCLI:
    """Command-line interface for dataset management."""

    def __init__(self, dataset_path: str = "data/processed/train_data.json"):
        self.dataset_path = Path(dataset_path)
        self.backup_dir = Path("data/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.data: List[Dict] = []

    def load_dataset(self):
        """Load dataset from JSON file."""
        if self.dataset_path.exists():
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = []
            self._error(f"Dataset not found: {self.dataset_path}")

    def save_dataset(self, backup: bool = True):
        """Save dataset to JSON file."""
        if backup and self.dataset_path.exists():
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{self.dataset_path.stem}_backup_{timestamp}.json"
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                backup_data = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(backup_data)

        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _output(self, data: any):
        """Output data as JSON."""
        print(json.dumps(data, ensure_ascii=False, indent=2))

    def _error(self, message: str, exit_code: int = 1):
        """Output error and exit."""
        print(json.dumps({"error": message}, ensure_ascii=False), file=sys.stderr)
        if exit_code > 0:
            sys.exit(exit_code)

    def _success(self, message: str, **kwargs):
        """Output success message."""
        result = {"success": True, "message": message}
        result.update(kwargs)
        self._output(result)

    def cmd_list(self, args):
        """List all samples with basic info."""
        samples = []
        for i, sample in enumerate(self.data):
            samples.append({
                "index": i,
                "question": sample["messages"][1]["content"][:150] + "..." if len(sample["messages"][1]["content"]) > 150 else sample["messages"][1]["content"],
                "source": sample["metadata"].get("source", ""),
                "title": sample["metadata"].get("title", "")
            })

        self._output({
            "total": len(self.data),
            "samples": samples
        })

    def cmd_get(self, args):
        """Get a specific sample by index."""
        index = args.index
        if not 0 <= index < len(self.data):
            self._error(f"Invalid index: {index}. Valid range: 0-{len(self.data)-1}")

        sample = self.data[index]
        self._output({
            "index": index,
            "question": sample["messages"][1]["content"],
            "answer": sample["messages"][2]["content"],
            "metadata": sample["metadata"]
        })

    def cmd_search(self, args):
        """Search samples."""
        query = args.query.lower()
        field = args.field
        results = []

        for i, sample in enumerate(self.data):
            question = sample["messages"][1]["content"]
            answer = sample["messages"][2]["content"]
            source = sample["metadata"].get("source", "")

            match = False
            if field == "question" and query in question.lower():
                match = True
            elif field == "answer" and query in answer.lower():
                match = True
            elif field == "source" and query in source.lower():
                match = True
            elif field == "all" and (query in question.lower() or query in answer.lower() or query in source.lower()):
                match = True

            if match:
                results.append({
                    "index": i,
                    "question": question[:150] + "..." if len(question) > 150 else question,
                    "answer": answer[:150] + "..." if len(answer) > 150 else answer,
                    "source": source
                })

        self._output({
            "query": args.query,
            "field": field,
            "total_results": len(results),
            "results": results
        })

    def cmd_add(self, args):
        """Add a new sample."""
        if not args.question or not args.answer:
            self._error("Question and answer are required")

        new_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un experto en regulación bancaria española. Tu tarea es responder preguntas sobre regulación bancaria, especialmente sobre parámetros de riesgo de crédito.\n\nIMPORTANTE:\n- Siempre cita la fuente de tu información\n- Si no estás seguro o no tienes la información, responde \"No tengo información suficiente para responder con certeza\" o \"Esta información no está disponible en los documentos proporcionados\"\n- Nunca inventes información\n- Sé preciso y específico en tus respuestas"
                },
                {
                    "role": "user",
                    "content": args.question
                },
                {
                    "role": "assistant",
                    "content": args.answer
                }
            ],
            "metadata": {
                "source": args.source or "CLI Entry",
                "url": args.url or "",
                "title": args.title or "CLI Entry"
            }
        }

        self.data.append(new_sample)
        self.save_dataset(backup=True)

        self._success(
            f"Added new sample",
            index=len(self.data) - 1,
            total_samples=len(self.data)
        )

    def cmd_update(self, args):
        """Update an existing sample."""
        index = args.index
        if not 0 <= index < len(self.data):
            self._error(f"Invalid index: {index}")

        if args.question:
            self.data[index]["messages"][1]["content"] = args.question
        if args.answer:
            self.data[index]["messages"][2]["content"] = args.answer
        if args.source:
            self.data[index]["metadata"]["source"] = args.source
        if args.url:
            self.data[index]["metadata"]["url"] = args.url
        if args.title:
            self.data[index]["metadata"]["title"] = args.title

        self.save_dataset(backup=True)
        self._success(f"Updated sample #{index}")

    def cmd_delete(self, args):
        """Delete a sample."""
        index = args.index
        if not 0 <= index < len(self.data):
            self._error(f"Invalid index: {index}")

        deleted = self.data.pop(index)
        self.save_dataset(backup=True)

        self._success(
            f"Deleted sample #{index}",
            question=deleted["messages"][1]["content"][:100],
            remaining_samples=len(self.data)
        )

    def cmd_stats(self, args):
        """Get dataset statistics."""
        if not self.data:
            self._output({"total": 0, "message": "No data"})
            return

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

        self._output({
            "total_samples": len(self.data),
            "avg_question_length": avg_q,
            "avg_answer_length": avg_a,
            "sources": sources
        })

    def cmd_export(self, args):
        """Export dataset to file."""
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        self._success(
            f"Exported dataset to {output_path}",
            total_samples=len(self.data),
            output_file=str(output_path)
        )

    def cmd_import(self, args):
        """Import dataset from file."""
        input_path = Path(args.input)
        if not input_path.exists():
            self._error(f"Input file not found: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            imported_data = json.load(f)

        if args.append:
            self.data.extend(imported_data)
            action = "appended to"
        else:
            self.data = imported_data
            action = "replaced"

        self.save_dataset(backup=True)

        self._success(
            f"Imported {len(imported_data)} samples ({action} dataset)",
            total_samples=len(self.data),
            imported=len(imported_data)
        )

    def cmd_validate(self, args):
        """Validate dataset structure."""
        errors = []
        warnings = []

        for i, sample in enumerate(self.data):
            # Check structure
            if "messages" not in sample:
                errors.append(f"Sample {i}: Missing 'messages' field")
                continue

            if len(sample["messages"]) != 3:
                errors.append(f"Sample {i}: Expected 3 messages, got {len(sample['messages'])}")

            if "metadata" not in sample:
                warnings.append(f"Sample {i}: Missing 'metadata' field")
            else:
                if "source" not in sample["metadata"]:
                    warnings.append(f"Sample {i}: Missing 'source' in metadata")

            # Check content
            if sample["messages"][1]["role"] != "user":
                errors.append(f"Sample {i}: Second message should be 'user', got '{sample['messages'][1]['role']}'")

            if sample["messages"][2]["role"] != "assistant":
                errors.append(f"Sample {i}: Third message should be 'assistant', got '{sample['messages'][2]['role']}'")

            # Check for empty content
            if not sample["messages"][1]["content"].strip():
                errors.append(f"Sample {i}: Empty question")

            if not sample["messages"][2]["content"].strip():
                errors.append(f"Sample {i}: Empty answer")

        self._output({
            "valid": len(errors) == 0,
            "total_samples": len(self.data),
            "errors": errors,
            "warnings": warnings
        })


def main():
    parser = argparse.ArgumentParser(
        description="RegLLM Dataset Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset',
        default='data/processed/train_data.json',
        help='Path to dataset file (default: data/processed/train_data.json)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # List command
    subparsers.add_parser('list', help='List all samples')

    # Get command
    get_parser = subparsers.add_parser('get', help='Get a specific sample')
    get_parser.add_argument('index', type=int, help='Sample index')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search samples')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument(
        '--field',
        choices=['all', 'question', 'answer', 'source'],
        default='all',
        help='Field to search in'
    )

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new sample')
    add_parser.add_argument('--question', required=True, help='Question text')
    add_parser.add_argument('--answer', required=True, help='Answer text')
    add_parser.add_argument('--source', default='CLI Entry', help='Source name')
    add_parser.add_argument('--url', default='', help='Source URL')
    add_parser.add_argument('--title', default='', help='Document title')

    # Update command
    update_parser = subparsers.add_parser('update', help='Update an existing sample')
    update_parser.add_argument('index', type=int, help='Sample index')
    update_parser.add_argument('--question', help='New question text')
    update_parser.add_argument('--answer', help='New answer text')
    update_parser.add_argument('--source', help='New source name')
    update_parser.add_argument('--url', help='New source URL')
    update_parser.add_argument('--title', help='New document title')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a sample')
    delete_parser.add_argument('index', type=int, help='Sample index')

    # Stats command
    subparsers.add_parser('stats', help='Get dataset statistics')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export dataset to file')
    export_parser.add_argument('--output', required=True, help='Output file path')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import dataset from file')
    import_parser.add_argument('--input', required=True, help='Input file path')
    import_parser.add_argument('--append', action='store_true', help='Append to existing dataset')

    # Validate command
    subparsers.add_parser('validate', help='Validate dataset structure')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize CLI
    cli = DatasetManagerCLI(dataset_path=args.dataset)
    cli.load_dataset()

    # Execute command
    command_method = getattr(cli, f'cmd_{args.command}', None)
    if command_method:
        command_method(args)
    else:
        cli._error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
