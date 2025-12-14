"""
Settings Manager for Lego Sorter Vision System

Handles persistent storage and retrieval of user settings including:
- Dataset selection (ldraw, ldview, rebrickable)
- Training parameters (epochs, batch_size)
- Camera configuration (camera_type)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_SETTINGS = {
    "dataset": "ldraw",
    "epochs": 10,
    "batch_size": 8,
    "camera_type": "usb"
}

class SettingsManager:
    def __init__(self, settings_path: str = None):
        """
        Initialize the settings manager.

        Args:
            settings_path: Path to settings JSON file. If None, uses default location.
        """
        if settings_path is None:
            # Default to output/settings.json
            base_path = Path("/app/output") if Path("/app/output").exists() else Path(__file__).parent.parent / "output"
            base_path.mkdir(parents=True, exist_ok=True)
            settings_path = base_path / "settings.json"

        self.settings_path = Path(settings_path)
        self.settings = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file or return defaults."""
        if self.settings_path.exists():
            try:
                with open(self.settings_path, 'r') as f:
                    settings = json.load(f)
                    logger.info(f"Loaded settings from {self.settings_path}")
                    # Merge with defaults to ensure all keys exist
                    return {**DEFAULT_SETTINGS, **settings}
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")
                return DEFAULT_SETTINGS.copy()
        else:
            logger.info("No settings file found, using defaults")
            return DEFAULT_SETTINGS.copy()

    def _save_settings(self) -> bool:
        """Save current settings to file."""
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.settings_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            logger.info(f"Saved settings to {self.settings_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        """Get all settings."""
        return self.settings.copy()

    def get(self, key: str, default=None) -> Any:
        """Get a specific setting."""
        return self.settings.get(key, default)

    def update(self, updates: Dict[str, Any]) -> bool:
        """
        Update settings with new values.

        Args:
            updates: Dictionary of settings to update

        Returns:
            True if saved successfully, False otherwise
        """
        self.settings.update(updates)
        return self._save_settings()

    def set(self, key: str, value: Any) -> bool:
        """
        Set a specific setting.

        Args:
            key: Setting key
            value: Setting value

        Returns:
            True if saved successfully, False otherwise
        """
        self.settings[key] = value
        return self._save_settings()

    def reset(self) -> bool:
        """Reset all settings to defaults."""
        self.settings = DEFAULT_SETTINGS.copy()
        return self._save_settings()


# Global settings manager instance
_settings_manager = None

def get_settings_manager() -> SettingsManager:
    """Get or create the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
