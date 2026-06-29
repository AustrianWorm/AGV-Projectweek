class ImageAnalyzationController:
    def __init__(self):
        self.image_analyzer = ImageAnalyzationController()

    def start_image_analysis(self, image):
        # Start the image analysis process
        result = self.image_analyzer.analyze_image(image)
        return result
    
    def stop_image_analysis(self):
        return