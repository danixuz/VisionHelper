//
//  ViewController.swift
//  VisionHelper
//
//  Created by דניאל שייך on 26/10/2024.
//

import UIKit
import AVKit
import Vision

enum mlModelType: String {
    case resnet50
    case resnet101
    case resnet152
    case mobilenetv2
    case yolov3
}

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    var bufferSize: CGSize = .zero
    var modelType: mlModelType = .yolov3

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        setUpCaptureSession()
        
    }
    
    private func setUpCaptureSession() {
        // Setting capture session and capture input to be video camera.
        let captureSession = AVCaptureSession()
//        captureSession.sessionPreset = .photo // optional for cropped look.
        let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)!
        do {
            captureSession.beginConfiguration()
            captureSession.sessionPreset = .vga640x480 // It’s important to choose the right resolution for your app. Don’t simply select the highest resolution available if your app doesn’t require it. It’s better to select a lower resolution so Vision can process results more efficiently.
            
            // Add input to the session.
            let input = try AVCaptureDeviceInput(device: videoDevice)
            captureSession.addInput(input)
        } catch {
            print("Couldn't set up capture session: \(error.localizedDescription).")
        }
        
        
        // Adding video output to capture session.
        let videoDataOutput = AVCaptureVideoDataOutput()
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)] // make sure to specify the pixel format.
            videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "captureVideoQueue"))
        } else {
            print("Couldn't add video data output to capture session.")
            captureSession.commitConfiguration()
            return
        }
        
        // Process every frame, but don’t hold on to more than one Vision request at a time.
        let captureConnection = videoDataOutput.connection(with: .video)
        captureConnection?.isEnabled = true
        do {
            try videoDevice.lockForConfiguration()
            let dimensions = CMVideoFormatDescriptionGetDimensions(videoDevice.activeFormat.formatDescription)
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            videoDevice.unlockForConfiguration()
        } catch {
            print("Error locking device for configuration: \(error.localizedDescription).")
        }
        
        captureSession.commitConfiguration()
        captureSession.startRunning()
        
        
        // Setting a preview layer for the capture session, and adding it to the view hierarchy.
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame
    }
    
    /// Analyze a frame from the capture session using it's CVPixelBuffer after capture.
    private func analyzeFrame(pixelBuffer: CVPixelBuffer) {
        if modelType == .mobilenetv2 {
            // Create and configure MobileNetV2 CoreML model.
            let modelConfig = MLModelConfiguration()
            guard let model = try? VNCoreMLModel(for: MobileNetV2.init(configuration: modelConfig).model) else {
                print("ERROR: Failed to create a CoreML model")
                return
            }
            
            // Perform request using CoreML model.
            let request = VNCoreMLRequest(model: model) { finishedReq, err in
                // check error?
                if let err = err {
                    print("ERROR: Failed to perform CoreML request: \(err)")
                    return
                }
                
                // Get observations from frame.
                guard let observations = finishedReq.results as? [VNClassificationObservation] else {
                    print("ERROR: No observations from CoreML request")
                    return
                }
                
                guard let firstObservation = observations.first else {
                    print("ERROR: No observations from CoreML request")
                    return
                }
                
                // Print detected object - what the camera thinks the object it's seeing is, and the confidence level.
                print("Object: \(firstObservation.identifier), Confidence: \(firstObservation.confidence)")
            }
            
            try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer).perform([request])
            
        } else if modelType == .yolov3 {
            // Create and configure YOLOV3
            let modelConfig = MLModelConfiguration()
            guard let model = try? VNCoreMLModel(for: YOLOv3.init(configuration: modelConfig).model) else {
                print("ERROR: Failed to create a CoreML model")
                return
            }
            let request = VNCoreMLRequest(model: model, completionHandler: { (finishedReq, error) in
                DispatchQueue.main.async(execute: {
                    // perform all the UI updates on the main queue
                    if let results = finishedReq.results {
                        self.parseYOLOV3RecognizedObjectObservations(results)
//                        print(results)
                    }
                    
                    // Print detected object - what the camera thinks the object it's seeing is, and the confidence level.
//                    print("Object: \(firstObservation), Confidence: \(firstObservation.confidence)")
                })
            })
            
            try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer).perform([request])
        }
            
        
    }
    
    private func parseYOLOV3RecognizedObjectObservations(_ results: [VNObservation]) {
        
        for observation in results where observation is VNRecognizedObjectObservation {
            guard let objectObservation = observation as? VNRecognizedObjectObservation else {
                continue
            }
            // Select only the label with the highest confidence.
            let topLabelObservation = objectObservation.labels[0]
            let objectBounds = VNImageRectForNormalizedRect(objectObservation.boundingBox, Int(bufferSize.width), Int(bufferSize.height))
            print("Object: \(topLabelObservation.identifier), Confidence: \(topLabelObservation.confidence)")
            
//            let textLayer = self.createTextSubLayerInBounds(objectBounds,
//                                                            identifier: topLabelObservation.identifier,
//                                                            confidence: topLabelObservation.confidence)
//            shapeLayer.addSublayer(textLayer)
//            detectionOverlay.addSublayer(shapeLayer)
        }
    }
    

    // - MARK: AVCaptureVideoDataOutputSampleBufferDelegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Error: No pixel buffer from sample buffer")
            return
        }
        
        analyzeFrame(pixelBuffer: pixelBuffer)
        
        
        
    }


}

