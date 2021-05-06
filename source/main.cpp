#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

constexpr std::string_view NAME_LABEL_FILE = "pascal-classes.txt"; // Tag file.
constexpr std::string_view NAME_DEPLOY_FILE = "fcn8s-heavy-pascal.prototxt"; // Description file.
constexpr std::string_view NAME_MODEL_FILE = "fcn8s-heavy-pascal.caffemodel"; // Training files.

constexpr std::string_view OUTPUT_NAME_FILE = "output.avi";

constexpr int WIDTH = 500;
constexpr int HEIGHT = 500;
constexpr int DELAY_MS = 100;

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

        file.close();
    }
}

int main()
{
    // Open the default video camera.
    cv::VideoCapture capture(cv::VideoCaptureAPIs::CAP_ANY);
    if (capture.isOpened() == false) {
        std::cerr << "Cannot open the video camera!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string path = std::filesystem::current_path().string() + '/';
    std::replace(path.begin(), path.end(), '\\', '/');

    const auto width = capture.get(cv::CAP_PROP_FRAME_WIDTH); // Get the width of frames of the video.
    const auto height = capture.get(cv::CAP_PROP_FRAME_HEIGHT); // Get the height of frames of the video.
    const auto fps = capture.get(cv::CAP_PROP_FPS);
    std::cout << "Resolution of the video: " << width << " x " << height << ".\nFrames per seconds: " << fps << "." << std::endl;

    std::vector<Label> labels;
    getLabelsFromFile(labels, path + NAME_LABEL_FILE.data());
    if (labels.empty()) {
        std::cerr << "Failed to read file!" << std::endl;
        return EXIT_FAILURE;
    }

    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
    cv::VideoWriter video(OUTPUT_NAME_FILE.data(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(WIDTH, HEIGHT));

    cv::Mat source;
    static constexpr int ESCAPE_KEY = 27;
    while (cv::waitKey(DELAY_MS) != ESCAPE_KEY) {
        // Read a new frame from video.
        if (capture.read(source) == false) { // Breaking the while loop if the frames cannot be captured.
            std::cerr << "Video camera is disconnected!" << std::endl;
            return EXIT_FAILURE;
        }
        resize(source, source, cv::Size(WIDTH, HEIGHT), 0, 0);

        cv::dnn::Net net;
        // Read binary file and description file.
        net = cv::dnn::readNetFromCaffe(path + NAME_DEPLOY_FILE.data(), path + NAME_MODEL_FILE.data());
        if (net.empty()) {
            std::cerr << "Could not load Caffe_net!" << std::endl;
            return EXIT_FAILURE;
        }

        const double start = cv::getTickCount();
        const cv::Mat blob = cv::dnn::blobFromImage(source);
        net.setInput(blob, "data");
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

        cv::Mat destination;
        cv::addWeighted(source, 0.3, result, 0.7, 0, destination); // Image merge.

        std::string name;
        for (const auto& index : indexes) {
            if (index != 0 && index < labels.size()) {
                name += labels[index].name + " & ";
            }
        }
        if (!name.empty()) {
            name.erase(name.end() - 3, name.end());
            cv::putText(destination, name, cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255), 1, 5);
        }
        cv::putText(destination, runTime, cv::Point(10, destination.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1, 5);
        const std::string resolution = std::to_string(destination.size().width) + "x" + std::to_string(destination.size().height);
        cv::putText(destination, resolution, cv::Point(destination.size().width - 80, destination.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1, 5);

        cv::imshow("FCN-demo", destination);

        // Write the frame into the file.
        video.write(destination);
    }

    capture.release();
    video.release();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
