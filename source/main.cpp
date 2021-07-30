#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include <boost/program_options.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

constexpr std::string_view NAME_LABEL_FILE = "pascal-classes.txt"; // Tag file
constexpr std::string_view NAME_DEPLOY_FILE = "fcn8s-heavy-pascal.prototxt"; // Description file
constexpr std::string_view NAME_MODEL_FILE = "fcn8s-heavy-pascal.caffemodel"; // Training files

constexpr int WIDTH = 500;
constexpr int HEIGHT = 500;
constexpr int DELAY_MS = 1;

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
            stream >> label.color[0];
            stream >> label.color[1];
            stream >> label.color[2];
            labels.push_back(std::move(label));
        }
        file.close();
    }
}

int main(int argc, char* argv[])
{
    std::string inputFile;
    std::string outputFile;
    bool useCuda;
    uint16_t frameNumber;
    boost::program_options::options_description desc("Options");
    desc.add_options()
        // All options:
        ("in,i", boost::program_options::value<std::string>(&inputFile), "Path to input file.\n") //
        ("out,o", boost::program_options::value<std::string>(&outputFile), "Path to output file.\n") //
        ("cuda,c", boost::program_options::value<bool>(&useCuda)->default_value(true), "Set CUDA enable.\n") //
        ("frame,f", boost::program_options::value<uint16_t>(&frameNumber)->default_value(1), "Set frame number.") //
        ("help,h", "Produce help message."); // Help
    boost::program_options::variables_map options;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), options);
        boost::program_options::notify(options);
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
    if (options.count("help")) {
        std::cout << desc << std::endl;
        return EXIT_SUCCESS;
    }

    cv::VideoCapture capture;
    if (inputFile.empty()) {
        // Open default video camera
        capture.open(cv::VideoCaptureAPIs::CAP_ANY);
    } else {
        capture.open(inputFile);
    }
    if (capture.isOpened() == false) {
        std::cerr << "Cannot open video!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string path = std::filesystem::current_path().string() + '/';
    std::replace(path.begin(), path.end(), '\\', '/');

    const auto width = capture.get(cv::CAP_PROP_FRAME_WIDTH); // Get width of frames of video
    const auto height = capture.get(cv::CAP_PROP_FRAME_HEIGHT); // Get height of frames of video
    const auto fps = capture.get(cv::CAP_PROP_FPS);
    std::cout << "Resolution of video: " << width << " x " << height << ".\nFrames per seconds: " << fps << "." << std::endl;

    std::vector<Label> labels;
    getLabelsFromFile(labels, path + NAME_LABEL_FILE.data());
    if (labels.empty()) {
        std::cerr << "Failed to read file!" << std::endl;
        return EXIT_FAILURE;
    }

    // Define codec and create VideoWriter object.output is stored in 'outcpp.avi' file
    cv::VideoWriter video;
    if (!outputFile.empty()) {
        video.open(outputFile + ".mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(WIDTH, HEIGHT));
    }

    bool cudaEnable = false;
    if (cv::cuda::getCudaEnabledDeviceCount() != 0) {
        cv::cuda::DeviceInfo deviceInfo;
        if (deviceInfo.isCompatible() != 0 && useCuda) {
            cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
            cudaEnable = true;
        }
    }

    static constexpr int ESCAPE_KEY = 27;
    while (cv::waitKey(DELAY_MS) != ESCAPE_KEY) {
        // Read a new frame from video.
        cv::Mat source;
        uint16_t i = 0;
        do {
            if (capture.read(source) == false) {
                std::cerr << "Video camera is disconnected!" << std::endl;
                return EXIT_FAILURE;
            }
            ++i;
        } while (i < frameNumber);
        resize(source, source, cv::Size(WIDTH, HEIGHT), 0, 0);

        cv::dnn::Net neuralNetwork;
        // Read binary file and description file
        neuralNetwork = cv::dnn::readNetFromCaffe(path + NAME_DEPLOY_FILE.data(), path + NAME_MODEL_FILE.data());
        if (neuralNetwork.empty()) {
            std::cerr << "Could not load Caffe_net!" << std::endl;
            return EXIT_FAILURE;
        }

        // Set CUDA as preferable backend and target
        if (cudaEnable) {
            neuralNetwork.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            neuralNetwork.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }

        const auto startTime = cv::getTickCount();
        const cv::Mat blob = cv::dnn::blobFromImage(source);
        neuralNetwork.setInput(blob, "data");
        cv::Mat score = neuralNetwork.forward("score");
        std::string runTime = "run time: " + std::to_string(static_cast<double>(cv::getTickCount() - startTime) / cv::getTickFrequency());
        runTime.erase(runTime.end() - 3, runTime.end());
        runTime += "s";

        const int rows = score.size[2]; // Image height
        const int cols = score.size[3]; // Width of image
        const int chns = score.size[1]; // Number of image channels
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

        // Look up colors
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
        cv::addWeighted(source, 0.3, result, 1.8, 0, destination); // Image merge

        std::string name;
        for (const auto& index : indexes) {
            if (index != 0 && index < labels.size()) {
                if (!labels[index].name.empty()) {
                    name += labels[index].name + " & ";
                }
            }
        }
        if (!name.empty()) {
            name.erase(name.end() - 3, name.end());
            cv::putText(destination, name, cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.1, cv::Scalar(0, 0, 255), 1, 5);
        }
        cv::putText(destination, runTime, cv::Point(10, destination.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
#ifdef NDEBUG
        cv::putText(destination, "in release", cv::Point(180, destination.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
#else
        cv::putText(destination, "in debug", cv::Point(180, destination.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
#endif
        if (cudaEnable) {
            cv::putText(destination, "using GPUs", cv::Point(300, destination.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
        } else {
            cv::putText(destination, "using CPUs", cv::Point(300, destination.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
        }
        const std::string resolution = std::to_string(destination.size().width) + "x" + std::to_string(destination.size().height);
        cv::putText(destination, resolution, cv::Point(destination.size().width - 80, destination.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);

        cv::imshow("FCN-demo", destination);

        // Write frame into file
        video.write(destination);
    }

    capture.release();
    video.release();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
