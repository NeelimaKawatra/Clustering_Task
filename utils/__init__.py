# utils/__init__.py - Updated with unified reset system
"""
Utility modules for the Clustery application with unified reset system
"""

# Import key functions that might be used across the application
from .session_state import (
    initialize_session_state,
    reset_analysis,
    cascade_from_data_loading,
    cascade_from_preprocessing,
    detect_changes_and_cascade,
    check_automatic_completion
)

from .reset_manager import (
    ResetManager,
    reset_full_analysis,
    reset_from_file_change,
    reset_from_column_change,
    reset_from_preprocessing_change,
    reset_from_clustering_change
)

from .helpers import (
    get_file_from_upload,
    validate_file_type,
    format_file_size,
    safe_column_name,
    preview_text,
    count_words,
    create_download_filename
)

from .styles import apply_custom_styles

__all__ = [
    # Session state functions
    'initialize_session_state',
    'reset_analysis', 
    'cascade_from_data_loading',
    'cascade_from_preprocessing',
    'detect_changes_and_cascade',
    'check_automatic_completion',
    
    # Unified reset system
    'ResetManager',
    'reset_full_analysis',
    'reset_from_file_change',
    'reset_from_column_change',
    'reset_from_preprocessing_change',
    'reset_from_clustering_change',
    
    # Helper functions
    'get_file_from_upload',
    'validate_file_type',
    'format_file_size',
    'safe_column_name',
    'preview_text',
    'count_words',
    'create_download_filename',
    
    # Styling
    'apply_custom_styles'
]