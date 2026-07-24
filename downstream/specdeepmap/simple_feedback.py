"""
Simple feedback class to replace QgsProcessingFeedback for standalone Python execution.
This allows the trainer and tester to work without QGIS dependencies.
"""


class SimpleFeedback:
    """Simple feedback class that mimics QgsProcessingFeedback interface."""
    
    def __init__(self):
        self._canceled = False
        self._progress = 0.0
    
    def pushInfo(self, message: str):
        """Print an info message."""
        print(f"[INFO] {message}")
    
    def pushWarning(self, message: str):
        """Print a warning message."""
        print(f"[WARNING] {message}")
    
    def pushError(self, message: str):
        """Print an error message."""
        print(f"[ERROR] {message}")
    
    def setProgress(self, progress: float):
        """Set progress percentage (0-100)."""
        self._progress = progress
        if int(progress) % 10 == 0:  # Print every 10% to avoid spam
            print(f"Progress: {progress:.1f}%")
    
    def isCanceled(self) -> bool:
        """Check if the process was canceled."""
        return self._canceled
    
    def cancel(self):
        """Cancel the process."""
        self._canceled = True
        print("[INFO] Process canceled by user.")
    
    def setProgressText(self, text: str):
        """Set progress text."""
        print(f"[PROGRESS] {text}")
