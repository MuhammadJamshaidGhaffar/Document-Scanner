#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <math.h>
#include <filesystem>


using namespace std;
using namespace cv;
namespace fs = std::filesystem;

//----------------------------------------------------------
//--------------- CONSTANTS --------------------------------
//----------------------------------------------------------
const float w = 250, h = 350;  // dimensions of our newly created scanned document
const Size resizeSize = Size(800, 640);
const std::string folderPath = "Resources/"; // a folder which contains images
const std::string scannedDocumentsFolderName = "scanned_documents/";

//----------------------------------------------------------
//---------------- FUNCTIONS -------------------------------
//----------------------------------------------------------
void createDirectories();
void selectImage();
void prepareImage(Mat inputImg, Mat &outputImg); // it will prepare the image for edge detection
vector<Point> getContour(Mat img );  // it will detect document and returns its corner points (4 corner points)  
double getDistance(Point origin, Point point); // returns distance between two points
Point shortestDistance(vector<Point> contour, Point origin); // returns the point which has shortest distance
void mouseCallBackFunc(int event, int x, int y, int flags, void* userData);	// mouse callback function on image													// from origin
bool isPointinCircle(Point centre, int radius, Point point); // checks if point is in circle
void saveImage(Mat img, string path="", string fileName="");


//----------------------------------------------------------
//--------------- GLOBAL VARIABLES -------------------------
//----------------------------------------------------------
String imgPath ;
Mat img ;
Point mouseCoords = Point(0, 0); // desc below
// these mouse coords will be updated continuously and will be used inside main function for drawing on screen
// checking if mouse is on some point , rect , or inside circle

Rect button; // this is a button Rect coordinates which will be displayed on top right corner
bool ok = false; // flag which tells that is ok button pressed

#define buttonWidth 70
#define buttonHeight 40



void main() {
	createDirectories();
	//------ Selecting Image -------------------
	selectImage();
	img = imread(imgPath);
	// matrix contains Perspective data from getPerpectiveTransfom
	Mat matrix, dilImg;
	// resizing image because sometimes image is too big
	resize(img, img, resizeSize);
	Size imgSize = img.size();
	// Defining Rect cordinates for button
	button = Rect(Point(imgSize.width - buttonWidth, 0), Point(imgSize.width , buttonHeight));

	// getting prepared image for edge detection
	prepareImage(img, dilImg);
	// gettting contours of detected document
	vector<Point> contour = getContour(dilImg );
	// go in infinite loop till ok button is not pressed
	while (!ok) {
		// here i am reading image again because i want to update the rectagle of detected document by mouse
		// click so first i will reload image again and then print rectangle again so previous rectangle will
		// be removed
		img = imread(imgPath);
		resize(img, img, resizeSize);
		// Displaying Ok button
		rectangle(img, button, Scalar(0, 0, 255), FILLED);
		putText(img, "Ok", Point(imgSize.width-buttonWidth+10,buttonHeight-10  ), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 2);
		// displaying rectangle around detected image
		for (int i = 0; i < 4; i++) {
			line(img, contour[i], contour[(i + 1) % 4], Scalar(255, 0, 255), 2);
		}
		// displaying circles on corner points
		for (int i = 0; i < 4; i++) {
			// if our mouse is on corner point change its colour else default colour
			if(isPointinCircle(contour[i] , 4 , mouseCoords))
				circle(img, contour[i], 3, Scalar(0, 0, 255), 2);
			else
				circle(img, contour[i], 3, Scalar(0, 255, 0), 2);
		}
		// Now display Images
		imshow("Image", img);
		setMouseCallback("Image", mouseCallBackFunc, &contour);
		waitKey(1);
		
	}
	destroyWindow("Image");
	//reading image again to remove all printed stuff on it
	img = imread(imgPath);
	resize(img, img, resizeSize);
	// getting the required image
	Point2f src[4]; // = { (Point2f)contour[0]  ,(Point2f)contour[1] ,(Point2f)contour[2] ,(Point2f)contour[3] };
	for (int i = 0; i < 4; i++) {
		src[i] = Point2f(contour[i].x, contour[i].y);
	}
	Point2f dest[4] = { {0.0f,0.0f}  , {w,0} , {w,h} , {0,h} };
	matrix = getPerspectiveTransform(src , dest);
	warpPerspective(img, img, matrix, Point(w, h));
	// displaying the image
	saveImage(img);
	imshow("Document", img);
	waitKey(0);

}



//----------------------------------------------------------
//---------------- FUNCTIONS -------------------------------
//----------------------------------------------------------


void selectImage() {
	vector<string> files;
	int counter = 0;
	for (const auto& entry : fs::directory_iterator(folderPath))
	{
		counter++;
		files.push_back(entry.path().string());
		std::cout << " (" << counter << ")  " << entry.path() << std::endl;
	}
	cout << "\nInput Image no (e.g: 1) : " << endl;
	int imgNo = -1;
	cin >> imgNo;
	imgPath = files[imgNo - 1];
	cout << "File selected : " << imgPath << endl;
}

void prepareImage(Mat inputImg, Mat &outputImg) {
	//Mat imgWarp, grayImg, blurImg, cannyImg;
	cvtColor(inputImg, outputImg, COLOR_BGR2GRAY);
	GaussianBlur(outputImg, outputImg, Size(3, 3), 3, 0);
	Canny(outputImg, outputImg, 25, 75);
	dilate(outputImg , outputImg, getStructuringElement(MORPH_RECT, Size(3, 3)));
}

vector<Point> getContour(Mat img ) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	double highestArea = 0;
	vector<Point> boundRect(4); // it will be rectangle which will store four points
	// format : top-left , top-right , bottom-right , bottom-left

	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		// here i am selecting that figure which have highest area and i will return its contours
		if (area > highestArea)
		{
			// get approximate corner points of detect figure
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], contours[i], 0.02 * peri, true);
			cout << "\nNo of points : " << contours[i].size() << endl;
			// if corner points are less than 4 it means it's either a triangle or straight line so skip
			if (contours[i].size() < 4)
				continue;

			highestArea = area;
			// store in boundRect , point of current contour which have shortest distance from origin
			boundRect[0] = shortestDistance(contours[i], Point(0, 0));
			boundRect[1] = shortestDistance(contours[i], Point(img.rows, 0));
			boundRect[2] = shortestDistance(contours[i], Point(img.rows, img.cols));
			boundRect[3] = shortestDistance(contours[i], Point(0, img.cols));

			// uncomment this if you want to debug
			// it will print current contours and then print boundRect ( our desired rect)
			/*printf("{ ");
			for (int j= 0; j < 4; j++) {

				printf("{%d , %d} , ", contours[i][j].x, contours[i][j].y);
			}
			printf("}");
			printf("\nBound Rect = { ");
			for (int i = 0; i < 4; i++) {

				printf("{%d , %d} , ", boundRect[i].x, boundRect[i].y);
			}
			printf("}");*/
			
		}
	}

	//-------- DEBUG -------------------
	// it will print final selected 
	/*printf("\nFinal Bound Rect = { ");
	for (int i = 0; i < 4; i++) {

		printf("{%d , %d} , ", boundRect[i].x, boundRect[i].y);
	}
	printf("}");*/
	return boundRect;
}

double getDistance(Point origin , Point point) {
	return sqrt( pow(point.x - origin.x ,2 ) + pow(point.y - origin.y , 2) );
}

Point shortestDistance(vector<Point> contour, Point origin) {
	int index = 0;
	double shortestDistance = getDistance(contour[0], origin);
	for (int i = 1; i < contour.size(); i++) {
		float distance = getDistance(contour[i], origin);
		if (distance < shortestDistance) {
			shortestDistance = distance;
			index = i;
		}
	}
	return contour[index];
}

void mouseCallBackFunc(int event, int x, int y, int flags, void* userData)
{
	static int selectedPointIndex = -1;

	mouseCoords = Point(x, y);
	vector<Point>* circlePoints = (vector<Point>*) userData;
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked Down - position (" << x << ", " << y << ")" << endl;
		if (button.contains(Point(x, y))) {
			ok = true;
		}
		for (int i = 0; i < circlePoints->size(); i++)
		{
			if (isPointinCircle(circlePoints->at(i), 10, Point(x, y))) {
				selectedPointIndex = i;
				cout << "Point no "<< selectedPointIndex << " selected" << endl;
				break;
			}
		}

	}
	else if (event == EVENT_LBUTTONUP)
	{
		selectedPointIndex = -1;
		cout << "Left button of the mouse is clicked Up - position (" << x << ", " << y << ")" << endl;
	}
	/*else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}*/
	else if (event == EVENT_MOUSEMOVE)
	{
		//cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
		if (selectedPointIndex != -1)
		{
			circlePoints->at(selectedPointIndex) = Point(x, y);
			cout << "Point no "<< selectedPointIndex << " has moved to ( "<<x << " , " << y << " )" << endl;
		}
	}
}

bool isPointinCircle(Point centre, int radius, Point point) {
	double ans = pow(point.x - centre.x, 2) + pow(point.y - centre.y, 2) - pow(radius, 2);
	if (ans < 0)
		return 1;
	else
		return 0;
}

void saveImage(Mat img , string path, string fileName) {
	if (path.length() == 0)
		path = folderPath;
	if (fileName.length() == 0)
		fileName = scannedDocumentsFolderName + to_string(time(0)) + ".jpg";
	imwrite(fileName, img);
}
void createDirectories() {
	fs::create_directory(folderPath);
	fs::create_directory(scannedDocumentsFolderName);
}