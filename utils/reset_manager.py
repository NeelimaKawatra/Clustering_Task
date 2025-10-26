# utils/reset_manager.py - Unified Reset System for Clustery
import streamlit as st
import time
from typing import List, Optional, Dict, Any

class ResetManager:
    """
    Unified reset management system for Clustery application.
    Handles all reset scenarios with configurable parameters.
    """
    
    def __init__(self):
        self.reset_levels = {
            'file_only': ['df', 'previous_file_key', 'uploaded_filename'],
            'data_loading': ['df', 'previous_file_key', 'uploaded_filename', 'tab_data_loading_complete'],
            'preprocessing': ['processed_texts', 'preprocessing_metadata', 'preprocessing_settings', 
                            'row_alignment', 'tab_preprocessing_complete', 'original_texts', 'preprocessing_tracked'],
            'clustering': ['clustering_results', 'tab_clustering_complete'],
            'finetuning': ['finetuning_results', 'finetuning_initialized', 'finetuning_ever_visited'],
            'column_selections': ['subjectID', 'entry_column', 'user_selections'],
            'navigation': ['current_page'],
            'session': ['session_id', 'backend'],
            'ui_state': ['file_uploader_key', 'file_uploader_reset', 'file_reset_reason', 'data_loading_alerts']
        }
    
    def _clear_column_choices(self):
        # Remove committed fields and the new draft→apply buckets
        for k in ["subjectID", "entry_column", "config", "temp"]:
            if k in st.session_state:
                del st.session_state[k]
        # Mark this tab incomplete
        st.session_state["tab_data_loading_complete"] = False

    
    def unified_reset(self, 
                     reset_type: str = "full",
                     preserve_columns: bool = False,
                     preserve_navigation: bool = True,
                     preserve_session: bool = True,
                     cascade_downstream: bool = True,
                     trigger_reason: str = "manual",
                     show_message: bool = True,
                     custom_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Unified reset function that handles all reset scenarios.
        
        Args:
            reset_type: Type of reset ('full', 'file_change', 'column_change', 'preprocessing_change', 'clustering_change')
            preserve_columns: Whether to preserve column selections
            preserve_navigation: Whether to preserve current page
            preserve_session: Whether to preserve session data
            cascade_downstream: Whether to reset downstream dependencies
            trigger_reason: Reason for reset (for logging)
            show_message: Whether to show user feedback
            custom_keys: Additional custom keys to reset
            
        Returns:
            Dictionary with reset summary
        """
        
        reset_summary = {
            'reset_type': reset_type,
            'keys_reset': [],
            'steps_affected': [],
            'preserved': [],
            'timestamp': time.time()
        }
        
        # Determine what to reset based on type
        if reset_type == "full":
            keys_to_reset = self._get_full_reset_keys(preserve_columns, preserve_navigation, preserve_session)
        elif reset_type == "file_change":
            keys_to_reset = self._get_file_change_keys(cascade_downstream)
        elif reset_type == "column_change":
            keys_to_reset = self._get_column_change_keys()
        elif reset_type == "preprocessing_change":
            keys_to_reset = self._get_preprocessing_change_keys()
        elif reset_type == "clustering_change":
            keys_to_reset = self._get_clustering_change_keys()
        else:
            keys_to_reset = custom_keys or []
        
        # Add custom keys if provided
        if custom_keys:
            keys_to_reset.extend(custom_keys)
        
        # Perform the reset
        reset_summary = self._execute_reset(keys_to_reset, reset_summary)
        
        # Handle special cases
        self._handle_special_resets(reset_type, reset_summary)
        
        # Update permanent progress
        self._update_permanent_progress(reset_type, reset_summary)
        
        # Log the reset
        self._log_reset(reset_summary, trigger_reason)
        
        # Show user feedback
        if show_message:
            self._show_reset_message(reset_summary, trigger_reason)
        
        return reset_summary
    
    def _get_full_reset_keys(self, preserve_columns: bool, preserve_navigation: bool, preserve_session: bool) -> List[str]:
        """Get keys for full application reset"""
        keys = []
        
        # Always reset these for full reset
        keys.extend(self.reset_levels['data_loading'])
        keys.extend(self.reset_levels['preprocessing'])
        keys.extend(self.reset_levels['clustering'])
        keys.extend(self.reset_levels['finetuning'])
        keys.extend(self.reset_levels['ui_state'])
        
        # Conditionally reset based on preserve flags
        if not preserve_columns:
            keys.extend(self.reset_levels['column_selections'])
        
        if not preserve_navigation:
            keys.extend(self.reset_levels['navigation'])
        
        if not preserve_session:
            keys.extend(self.reset_levels['session'])
        
        return list(set(keys))  # Remove duplicates
    
    def _get_file_change_keys(self, cascade_downstream: bool) -> List[str]:
        """Get keys for file change reset"""
        keys = self.reset_levels['file_only'].copy()
        
        if cascade_downstream:
            keys.extend(self.reset_levels['data_loading'])
            keys.extend(self.reset_levels['preprocessing'])
            keys.extend(self.reset_levels['clustering'])
            keys.extend(self.reset_levels['finetuning'])
        
        return keys
    
    def _get_column_change_keys(self) -> List[str]:
        """Get keys for column selection change reset"""
        keys = []
        keys.extend(self.reset_levels['preprocessing'])
        keys.extend(self.reset_levels['clustering'])
        keys.extend(self.reset_levels['finetuning'])
        return keys
    
    def _get_preprocessing_change_keys(self) -> List[str]:
        """Get keys for preprocessing change reset"""
        keys = []
        keys.extend(self.reset_levels['clustering'])
        keys.extend(self.reset_levels['finetuning'])
        return keys
    
    def _get_clustering_change_keys(self) -> List[str]:
        """Get keys for clustering change reset"""
        return self.reset_levels['finetuning'].copy()
    
    def _execute_reset(self, keys_to_reset: List[str], reset_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual reset of session state keys"""
        
        for key in keys_to_reset:
            if key in st.session_state:
                old_value = st.session_state[key]
                
                # Handle special reset values
                if key == 'preprocessing_settings':
                    st.session_state[key] = {
                        'method': 'none',
                        'details': 'No preprocessing applied',
                        'custom_settings': {}
                    }
                elif key in ['original_texts', 'row_alignment']:
                    st.session_state[key] = []
                elif key == 'user_selections':
                    st.session_state[key] = {
                        'id_column_choice': None,
                        'entry_column_choice': None,
                        'original_columns': []
                    }
                #elif key == 'entry_column':
                #    st.session_state[key] = "-- Select an entry column --"
                #elif key == 'subjectID':
                #    st.session_state[key] = "-- Select a subject ID column--"
                elif key == 'current_page':
                    st.session_state[key] = "data_loading"
                elif key in ['data_loading_alerts']:
                    st.session_state[key] = []
                elif key.endswith('_complete') or key.endswith('_tracked') or key.endswith('_initialized') or key.endswith('_visited'):
                    st.session_state[key] = False
                else:
                    st.session_state[key] = None
                
                reset_summary['keys_reset'].append(key)
                
                # Track what steps were affected
                if 'data_loading' in key or key in ['df', 'subjectID', 'entry_column']:
                    if 'data_loading' not in reset_summary['steps_affected']:
                        reset_summary['steps_affected'].append('data_loading')
                
                if 'preprocessing' in key or key in ['processed_texts', 'original_texts']:
                    if 'preprocessing' not in reset_summary['steps_affected']:
                        reset_summary['steps_affected'].append('preprocessing')
                
                if 'clustering' in key:
                    if 'clustering' not in reset_summary['steps_affected']:
                        reset_summary['steps_affected'].append('clustering')
                
                if 'finetuning' in key:
                    if 'finetuning' not in reset_summary['steps_affected']:
                        reset_summary['steps_affected'].append('finetuning')
        
        return reset_summary
    
    def _handle_special_resets(self, reset_type: str, reset_summary: Dict[str, Any]):
        """Handle special reset cases"""
        
        # File uploader reset
        if reset_type in ['full', 'file_change']:
            st.session_state.file_uploader_key = f"uploader_{int(time.time())}"
            st.session_state.file_uploader_reset = True
            st.session_state.file_reset_reason = reset_type
            st.session_state["data_loading_alerts"] = []
        
        # Clear state fingerprints for change detection
        if reset_type in ['full', 'file_change', 'column_change']:
            st.session_state.state_fingerprints = {}
        
        # Clear method tracking for preprocessing
        if reset_type in ['full', 'file_change', 'column_change', 'preprocessing_change']:
            if 'last_preprocessing_method' in st.session_state:
                del st.session_state['last_preprocessing_method']
        
        # Clear backend caches
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass

        # Clear column choices + draft/commit buckets on key reset types
        if reset_type in ['full', 'file_change']:
            self._clear_column_choices()
        
        # Reset finetuning backend instance when finetuning is reset
        if reset_type in ['full', 'file_change', 'column_change', 'preprocessing_change', 'clustering_change']:
            if 'finetuning_backend_instance' in st.session_state:
                try:
                    st.session_state.finetuning_backend_instance.reset()
                except AttributeError:
                    # Backend doesn't have reset method, just delete it
                    del st.session_state['finetuning_backend_instance']
                except Exception:
                    pass

    
    def _update_permanent_progress(self, reset_type: str, reset_summary: Dict[str, Any]):
        """Update permanent progress tracking"""
        
        if 'permanent_progress' not in st.session_state:
            st.session_state.permanent_progress = {
                'data_loading': False,
                'preprocessing': False,
                'clustering': False
            }
        
        # Reset permanent progress based on what was affected
        if reset_type == 'full':
            st.session_state.permanent_progress = {
                'data_loading': False,
                'preprocessing': False,
                'clustering': False
            }
        elif 'data_loading' in reset_summary['steps_affected']:
            st.session_state.permanent_progress['data_loading'] = False
            st.session_state.permanent_progress['preprocessing'] = False
            st.session_state.permanent_progress['clustering'] = False
        elif 'preprocessing' in reset_summary['steps_affected']:
            st.session_state.permanent_progress['preprocessing'] = False
            st.session_state.permanent_progress['clustering'] = False
        elif 'clustering' in reset_summary['steps_affected']:
            st.session_state.permanent_progress['clustering'] = False
    

    def _log_reset(self, reset_summary: Dict[str, Any], trigger_reason: str):
        """Log the reset for analytics"""
        
        if hasattr(st.session_state, 'backend') and st.session_state.backend:
            try:
                st.session_state.backend.track_activity(
                    st.session_state.get('session_id', 'unknown'), 
                    "unified_reset", 
                    {
                        "reset_type": reset_summary['reset_type'],
                        "trigger_reason": trigger_reason,
                        "steps_affected": reset_summary['steps_affected'],
                        "keys_reset_count": len(reset_summary['keys_reset']),
                        "timestamp": reset_summary['timestamp']
                    }
                )
            except Exception:
                pass  # Don't fail on logging errors
    
    
    def _show_reset_message(self, reset_summary: Dict[str, Any], trigger_reason: str):
        """Show appropriate user feedback"""
        
        steps_affected = reset_summary['steps_affected']
        
        if trigger_reason == "start_new_analysis":
            st.success("🔄 Analysis reset complete! Upload a new file to begin.")
        elif len(steps_affected) > 2:
            step_list = ", ".join(steps_affected)
            st.warning(f"🔄 Reset complete. Affected steps: {step_list}")
        elif len(steps_affected) == 1:
            step = steps_affected[0].replace('_', ' ').title()
            st.info(f"🔄 {step} reset. Please reconfigure this step.")
        elif len(steps_affected) > 0:
            st.info("🔄 Configuration updated. Please review affected steps.")

# Convenience functions for common reset scenarios
def reset_full_analysis(preserve_columns: bool = False, show_message: bool = True) -> Dict[str, Any]:
    """Complete application reset"""
    manager = ResetManager()
    return manager.unified_reset(
        reset_type="full",
        preserve_columns=preserve_columns,
        preserve_navigation=False,
        trigger_reason="start_new_analysis",
        show_message=show_message
    )

def reset_from_file_change(show_message: bool = True) -> Dict[str, Any]:
    """Reset when file changes"""
    manager = ResetManager()
    return manager.unified_reset(
        reset_type="file_change",
        preserve_columns=True,
        trigger_reason="file_change",
        show_message=show_message
    )

def reset_from_column_change(changed_column: str, show_message: bool = True) -> Dict[str, Any]:
    """Reset when column selection changes"""
    manager = ResetManager()
    return manager.unified_reset(
        reset_type="column_change",
        preserve_columns=True,
        trigger_reason=f"{changed_column}_column_change",
        show_message=show_message
    )

def reset_from_preprocessing_change(show_message: bool = True) -> Dict[str, Any]:
    """Reset when preprocessing settings change"""
    manager = ResetManager()
    return manager.unified_reset(
        reset_type="preprocessing_change",
        trigger_reason="preprocessing_change",
        show_message=show_message
    )

def reset_from_clustering_change(show_message: bool = True) -> Dict[str, Any]:
    """Reset when clustering parameters change"""
    manager = ResetManager()
    return manager.unified_reset(
        reset_type="clustering_change",
        trigger_reason="clustering_change",
        show_message=show_message
    )

# Backward compatibility functions
def reset_analysis():
    """Legacy function - now uses unified system"""
    return reset_full_analysis(preserve_columns=False, show_message=True)

def cascade_from_data_loading():
    """Legacy function - now uses unified system"""
    return reset_from_column_change("data_loading", show_message=True)

def cascade_from_preprocessing():
    """Legacy function - now uses unified system"""
    return reset_from_preprocessing_change(show_message=True)