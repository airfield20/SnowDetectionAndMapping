/**
 * @authors Aaron Cofield Ben Didonato
 */

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

class SnowDetector{
public:
    SnowDetector(){
        n = ros::NodeHandle();
        image_transport::ImageTransport it(n);
        image_sub = it.subscribe("/left/transformed_image", 1, &SnowDetector::image_callback, this);
        image_pub = it.advertise("/left/snow_overlay", 1);
    }

private:
    void publish_image(const cv::Mat& img, image_transport::Publisher &pubImg, const std::string& imgFrameId, const ros::Time& t){
        sensor_msgs::ImagePtr msg;
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", img).toImageMsg();
        msg->header.stamp = t;
        msg->header.frame_id = imgFrameId;
        pubImg.publish(msg);
    }
    void image_callback(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat snow, blurimage, origimage,screenshot;
        origimage = cv_ptr->image;
        screenshot = cv_ptr->image;

        cv::GaussianBlur(origimage, snow, cv::Size(5, 5), 0, 0);
        blurimage = snow;
        cv::inRange(snow, cv::Scalar(150, 150, 150), cv::Scalar(255, 255, 255), snow);
        cv::Mat ground_mask = cv::Mat::zeros(snow.size(), snow.type());
//        cv::rectangle(ground_mask, cv::Rect(0, ground_mask.rows , ground_mask.cols, ground_mask.rows), cv::Scalar(255, 255, 255), CV_FILLED);
//        cv::bitwise_and(snow, ground_mask, snow);

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
//            std::cout << i << "" << contours[i] << std::endl;
        }

        /// Draw polygonal contour + bonding rects + circles
        cv::Mat drawing = cv::Mat::zeros(threshold_output.size(), CV_8UC3);
        for (int i = 0; i < contours.size(); i++)
        {
            cv::Scalar color = cv::Scalar(255, 0, 255);
            drawContours(drawing, contours_poly, i, color, CV_FILLED, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
        }

        cv::namedWindow("Snow", cv::WINDOW_NORMAL);
        cv::resizeWindow("Snow", 1280, 720);
        double alpha = 0.5;
        double beta = 1 - alpha;
        cv::Mat overlayed;
        addWeighted(drawing,alpha,blurimage,beta,0,overlayed);
        cv::imshow("Snow", overlayed);
        cv::waitKey(1);
        cv_ptr->image = drawing;
        image_pub.publish(cv_ptr->toImageMsg());
    }

    ros::NodeHandle n;
    image_transport::Subscriber image_sub;
    image_transport::Publisher image_pub;

};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "snow_detector");
    SnowDetector sd;
    ros::spin();

}
