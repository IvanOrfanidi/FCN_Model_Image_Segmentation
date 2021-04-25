#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

const std::string_view NAME_LABEL_FILE = "pascal-classes.txt";
const std::string_view NAME_DEPLOY_FILE = "fcn8s-heavy-pascal.prototxt";
const std::string_view NAME_MODEL_FILE = "fcn8s-heavy-pascal.caffemodel";
constexpr std::string_view OUTPUT_NAME_FILE = "output.avi";

constexpr auto WIDTH = 500;
constexpr auto HEIGHT = 500;
constexpr auto DELAY_MS = 100;

struct Label {
    std::string name;
    cv::Vec3b color;
};

void getLabelsFromFile(std::vector<Label>& labels, const std::string& nameFile)
{
    std::ifstream file;
    file.open(nameFile, std::ifstream::in);
    if (file.is_open()) {
        while (!file.eof()) {
            std::string line;
            std::getline(file, line);
            std::stringstream stream(line);

            Label label;
            stream >> label.name;
            int color;
            stream >> color;
            label.color[0] = color;
            stream >> color;
            label.color[1] = color;
            stream >> color;
            label.color[2] = color;

            labels.push_back(label);
        }
    }
}

int main()
{
    // Open the default video camera.
    cv::VideoCapture cap(cv::VideoCaptureAPIs::CAP_ANY);
    if (cap.isOpened() == false) {
        std::cerr << "Cannot open the video camera!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string selfPath = std::filesystem::current_path().string() + '/';
    std::replace(selfPath.begin(), selfPath.end(), '\\', '/');

    const auto width = cap.get(cv::CAP_PROP_FRAME_WIDTH); // Get the width of frames of the video.
    const auto height = cap.get(cv::CAP_PROP_FRAME_HEIGHT); // Get the height of frames of the video.
    const auto fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Resolution of the video: " << width << " x " << height << ".\nFrames per seconds: " << fps << "." << std::endl;

    std::vector<Label> labels;
    getLabelsFromFile(labels, selfPath + NAME_LABEL_FILE.data());
    if (labels.empty()) {
        std::cerr << "Failed to read file!" << std::endl;
        return EXIT_FAILURE;
    }

    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
    cv::VideoWriter video(OUTPUT_NAME_FILE.data(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(WIDTH, HEIGHT));

    cv::Mat src;
    static constexpr int ESCAPE_KEY = 27;
    while (cv::waitKey(DELAY_MS) != ESCAPE_KEY) {
        // Read a new frame from video.
        if (cap.read(src) == false) { // Breaking the while loop if the frames cannot be captured.
            std::cerr << "Video camera is disconnected!" << std::endl;
            return EXIT_FAILURE;
        }
        resize(src, src, cv::Size(WIDTH, HEIGHT), 0, 0);

        cv::dnn::Net net;
        // Read binary file and description file.
        net = cv::dnn::readNetFromCaffe(selfPath + NAME_DEPLOY_FILE.data(), selfPath + NAME_MODEL_FILE.data());

        const double start = cv::getTickCount();
        net.setInput(cv::dnn::blobFromImage(src), "data");
        const cv::Mat score = net.forward("score");
        std::string runTime = "run time: " + std::to_string((cv::getTickCount() - start) / cv::getTickFrequency());
        runTime.erase(runTime.end() - 3, runTime.end());

        const int& rows = score.size[2]; // Image height.
        const int& cols = score.size[3]; // The width of the image.
        const int& chns = score.size[1]; // Number of image channels.
        cv::Mat maxCl(rows, cols, CV_8UC1);
        cv::Mat maxVal(rows, cols, CV_32FC1);

        for (int c = 0; c < chns; c++) {
            for (int row = 0; row < rows; row++) {
                const float* ptrScore = score.ptr<float>(0, c, row);
                uchar* ptrMaxCl = maxCl.ptr<uchar>(row);
                float* ptrMaxVal = maxVal.ptr<float>(row);
                for (int col = 0; col < cols; col++) {
                    if (ptrScore[col] > ptrMaxVal[col]) {
                        ptrMaxVal[col] = ptrScore[col];
                        ptrMaxCl[col] = static_cast<uchar>(c);
                    }
                }
            }
        }

        // Look up colors.
        std::set<size_t> indexes;
        cv::Mat result = cv::Mat::zeros(rows, cols, CV_8UC3);
        for (int row = 0; row < rows; row++) {
            const uchar* ptrMaxCl = maxCl.ptr<uchar>(row);
            cv::Vec3b* ptrColor = result.ptr<cv::Vec3b>(row);
            for (int col = 0; col < cols; col++) {
                const size_t index = ptrMaxCl[col];
                indexes.emplace(index);
                ptrColor[col] = labels[index].color;
            }
        }

        cv::Mat dst;
        cv::addWeighted(src, 0.3, result, 0.7, 0, dst); // Image merge.

        std::string name;
        for (const auto& index : indexes) {
            if (index != 0 && index < labels.size()) {
                name += labels[index].name + " & ";
            }
        }
        if (!name.empty()) {
            name.erase(name.end() - 3, name.end());
            cv::putText(dst, name, cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 1, 5);
        }
        cv::putText(dst, runTime, cv::Point(10, dst.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1, 5);
        const std::string resolution = std::to_string(dst.size().width) + "x" + std::to_string(dst.size().height);
        cv::putText(dst, resolution, cv::Point(dst.size().width - 80, dst.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1, 5);

        cv::imshow("FCN-demo", dst);

        // Write the frame into the file.
        video.write(dst);
    }

    cap.release();
    video.release();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
