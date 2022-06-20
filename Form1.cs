using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using Emgu.CV.CvEnum;
using System.IO;
using System.Threading;
using System.Diagnostics;

namespace FaceRecognitionApp
{
    public partial class Form1 : Form
    {
        #region Variables
        private Capture videoCapture = null;
        private Image<Bgr, Byte> currentFrame = null;
        Mat frame = new Mat();
        private bool facesDetectionEnable = false;
        CascadeClassifier faceCascadeClassifier = new CascadeClassifier(@"C:\Users\alaia\Documents\Visual Studio 2019\Projects\FaceRecognitionApp\haarcascade_frontalface_alt.xml");
        Image<Bgr, byte> faceResult = null;
        List<Image<Gray, Byte>> trainedFaces = new List<Image<Gray, byte>>();
        List<int> personsLabels = new List<int>();
        
        bool enableSaveImage = false;
        private bool isTrained = false;
        EigenFaceRecognizer recognizer;
        List<string> PersonsNames = new List<string>();

        #endregion
        public Form1()
        {
            InitializeComponent();
        }

        private void btnCapture_Click(object sender, EventArgs e)
        {
            if (videoCapture != null) videoCapture.Dispose();
            videoCapture = new Capture();
            Application.Idle += ProcessFrame;
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            //step 1: video capture
            if(videoCapture != null && videoCapture.Ptr != IntPtr.Zero)
            videoCapture.Retrieve(frame, 0);
            currentFrame = frame.ToImage<Bgr, Byte>().Resize(picCapture.Width, picCapture.Height, Inter.Cubic);

            //step 2: detect faces
            if(facesDetectionEnable)
            {
                //convert Bgr image to grayscale image
                Mat grayImage = new Mat();
                CvInvoke.CvtColor(currentFrame, grayImage, ColorConversion.Bgr2Gray);
                //Enhance the image to get better result
                CvInvoke.EqualizeHist(grayImage, grayImage);

                Rectangle[] faces = faceCascadeClassifier.DetectMultiScale(grayImage, 1.1, 3, Size.Empty, Size.Empty);
                //if faces detected
                if(faces.Length > 0)
                {
                    int faceID = 0;
                    foreach(var face in faces)
                    {
                        //draw square around each face 
                        CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Red).MCvScalar, 2);

                        //Step 3: Add Person
                        //assign the picture to the person
                        Image<Bgr, Byte> resultImage = currentFrame.Convert<Bgr, Byte>();
                        resultImage.ROI = face;
                        picDetected.SizeMode = PictureBoxSizeMode.StretchImage;
                        picDetected.Image = resultImage.Bitmap;

                        if(enableSaveImage)
                        {
                            //we will create a directory if it does not exist
                            string path = Directory.GetCurrentDirectory() + @"\TrainedImages";
                            if (!Directory.Exists(path))
                                Directory.CreateDirectory(path);

                            //we will save 10 images with delay of 1 second for each image
                            // to avoid GUI hangs we will use another task 
                            Task.Factory.StartNew(() => {
                                for (int i = 0; i < 10; i++)
                                {
                                    //resize the image then saving it
                                    resultImage.Resize(200, 200, Inter.Cubic).Save(path + @"\" + txtPersonName.Text + "_" + DateTime.Now.ToString("dd-mm-yyyy-hh-mm-ss") + ".jpg");
                                    Thread.Sleep(1000);
                                }
                            });

                            enableSaveImage = false;

                            if(btnAddPerson.InvokeRequired)
                            {
                                btnAddPerson.Invoke(new ThreadStart(delegate
                                {
                                    btnAddPerson.Enabled = true;
                                }));
                            }

                            //step 5: recognize the face
                            if(isTrained)
                            {
                                Image<Gray, Byte> grayFaceResult = resultImage.Convert<Gray, Byte>().Resize(200, 200, Inter.Cubic);
                                CvInvoke.EqualizeHist(grayFaceResult, grayFaceResult);
                                var result = recognizer.Predict(grayFaceResult);
                                pictureBox1.Image = grayFaceResult.Bitmap;
                                pictureBox2.Image = trainedFaces[result.Label].Bitmap;
                                Debug.WriteLine(result.Label + ". " + result.Distance);
                                //Here results found known faces
                                if (result.Label != -1 && result.Distance < 2000)
                                {
                                    CvInvoke.PutText(currentFrame, PersonsNames[result.Label], new Point(face.X - 2, face.Y - 2),
                                        FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                                    CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Green).MCvScalar, 2);
                                }
                                //here results did not found any know faces
                                else
                                {
                                    CvInvoke.PutText(currentFrame, "Unknown", new Point(face.X - 2, face.Y - 2),
                                        FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                                    CvInvoke.Rectangle(currentFrame, face, new Bgr(Color.Red).MCvScalar, 2);

                                }
                            }
                            else { Debug.WriteLine("picture is not trained"); }
                        }
                    }
                }
            }
            
            //render the video capture into the Picture box picCapture
            picCapture.Image = currentFrame.Bitmap;


        }

        private void btnDetectionFaces_Click(object sender, EventArgs e)
        {
            facesDetectionEnable = true;

        }

        private void btnAddPerson_Click(object sender, EventArgs e)
        {
            btnAddPerson.Enabled = false;
            enableSaveImage = true; 
        }


        private void btnTrain_Click(object sender, EventArgs e)
        {
            TrainImagesFromDisk();
        }

        private bool TrainImagesFromDisk()
        {
            int imagesCount = 0;
            double threshold = 2000;
            trainedFaces.Clear();
            personsLabels.Clear();
            PersonsNames.Clear();

            try
            {
                string path = Directory.GetCurrentDirectory() + @"\TrainedImages";
                string[] files = Directory.GetFiles(path, "*.jpg", SearchOption.AllDirectories);

                foreach (var file in files)
                {
                    Image<Gray, byte> trainedImage = new Image<Gray, byte>(file).Resize(200, 200, Inter.Cubic);
                    CvInvoke.EqualizeHist(trainedImage, trainedImage);
                    trainedFaces.Add(trainedImage);
                    personsLabels.Add(imagesCount);
                    string name = file.Split('\\').Last().Split('_')[0];
                    PersonsNames.Add(name);
                    imagesCount++;
                    Debug.WriteLine(imagesCount + ' ' + name);
                }

                if(trainedFaces.Count() > 0)
                {
                    recognizer = new EigenFaceRecognizer(imagesCount, threshold);
                    recognizer.Train(trainedFaces.ToArray(), personsLabels.ToArray());

                    isTrained = true;

                    return true;
                }
                else
                {
                    isTrained = false;
                    return false;  
                }
            }
            catch(Exception ex)
            {
                isTrained = false;
                MessageBox.Show("Error in train images: " + ex.Message);
                return false;
            }
        }
    }
}
