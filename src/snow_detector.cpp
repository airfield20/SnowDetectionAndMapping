#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::Mat screenshot;
    screenshot = cv::imread("Screenshot_2019-01-03_17-29-04.png");

    cv::Mat snow, blurimage, origimage;
    //cap >> origimage;
    origimage = screenshot;
    if (origimage.empty())
      return -1;

    cv::GaussianBlur(origimage, snow, cv::Size(5, 5), 0, 0);
    blurimage = snow;
    cv::inRange(snow, cv::Scalar(150, 150, 150), cv::Scalar(255, 255, 255), snow);
    cv::Mat ground_mask = cv::Mat::zeros(snow.size(), snow.type());
    cv::rectangle(ground_mask, cv::Rect(0, 2.5 * ground_mask.rows / 4, ground_mask.cols, ground_mask.rows / 4), cv::Scalar(255, 255, 255), CV_FILLED);
    cv::bitwise_and(snow, ground_mask, snow);

    int thresh = 100;
    int k_erode_size = 10;
    int k_dialate_size = 25;
    cv::RNG rng(12345);

    /* cv::createTrackbar("thresh", "Snow", &thresh, 255);
    cv::createTrackbar("erode", "Snow", &k_erode_size, 255);
    cv::createTrackbar("dialate", "Snow", &k_dialate_size, 255);
    */

    cv::Mat kernel_erode = cv::Mat::ones(k_erode_size, k_erode_size, CV_32F) / (float)(k_erode_size * k_erode_size);
    cv::Mat kernel_dialate = cv::Mat::ones(k_dialate_size, k_dialate_size, CV_32F) / (float)(k_dialate_size * k_dialate_size);
    cv::erode(snow, snow, kernel_erode);
    cv::dilate(snow, snow, kernel_dialate);
    cv::Mat threshold_output;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::threshold(snow, threshold_output, thresh, 255, cv::THRESH_BINARY);
    /// Find contours
    cv::findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    /// Approximate contours to polygons + get bounding rects and circles
    std::vector<std::vector<cv::Point>> contours_poly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());
    std::vector<cv::Point2f> center(contours.size());
    

    for (int i = 0; i < contours.size(); i++)
    {
      cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
      boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
      std::cout << i << "" << contours[i] << std::endl;
    }

    /// Draw polygonal contour + bonding rects + circles
    cv::Mat drawing = cv::Mat::zeros(threshold_output.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++)
    {
      cv::Scalar color = cv::Scalar(255, 0, 255);
      drawContours(drawing, contours_poly, i, color, CV_FILLED, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
      //rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
      
    }

    cv::Point Q1 = cv::Point2f(414, 450); //top left pixel coordinate
    cv::Point Q2 = cv::Point2f(878, 450); //top right
    cv::Point Q3 = cv::Point2f(1194, 625); //bottom right
    cv::Point Q4 = cv::Point2f(182, 625); //bottom left

    double ratio = 1.31578947; // width / height of the actual panel on the ground
    double cardH = sqrt((Q3.x - Q2.x) * (Q3.x - Q2.x) + (Q3.y - Q2.y) * (Q3.y - Q2.y));
    double cardW = ratio * cardH;

    cv::Rect R(Q1.x, Q1.y, cardW, cardH);

    cv::Point R1 = cv::Point2f(R.x, R.y);
    cv::Point R2 = cv::Point2f(R.x + R.width, R.y);
    cv::Point R3 = cv::Point2f(cv::Point2f(R.x + R.width, R.y + R.height));
    cv::Point R4 = cv::Point2f(cv::Point2f(R.x, R.y + R.height));

    std::vector<cv::Point2f> quad_pts{Q1, Q2, Q3, Q4};
    std::vector<cv::Point2f> squre_pts{R1, R2, R3, R4};

    cv::Mat transmtx = cv::getPerspectiveTransform(quad_pts, squre_pts);
    cv::Mat transformed = cv::Mat::zeros(screenshot.rows, screenshot.cols, CV_8UC3);

    cv::warpPerspective(screenshot, transformed, transmtx, transformed.size());
    

    cv::namedWindow("Snow", cv::WINDOW_NORMAL);
    cv::resizeWindow("Snow", 1280, 720);
    cv::imshow("Snow", (drawing | blurimage));

    /* cv::namedWindow("Depth", cv::WINDOW_NORMAL);
    cv::resizeWindow("Depth", 1280, 720);
    cv::imshow("Depth", (depth_view | drawing)); */

    cv::namedWindow("Warp", cv::WINDOW_NORMAL);
    cv::resizeWindow("Warp", 1280, 720);
    cv::imshow("Warp", transformed);

    while (1){
      char c = (char)cv::waitKey(1);
      if (c == 32)
        return (0);
    }
    
  
  //zed.close();
  
}
