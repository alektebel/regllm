#!/bin/bash

# Launch RegLLM Dataset Manager
# This script starts the web UI for managing the training dataset

echo "======================================================================"
echo "RegLLM - Dataset Manager"
echo "======================================================================"
echo ""
echo "Starting web interface..."
echo "Access at: http://localhost:7861"
echo ""
echo "Features:"
echo "  • Browse and search dataset"
echo "  • Add new Q&A samples"
echo "  • Edit existing samples"
echo "  • Delete samples"
echo "  • View statistics"
echo "  • Automatic backups"
echo ""
echo "Press Ctrl+C to stop"
echo "======================================================================"
echo ""

python dataset_manager_ui.py
