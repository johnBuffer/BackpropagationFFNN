#include <iostream>
#include <bitset>
#include "engine/window_context_handler.hpp"
#include "engine/common/binary_io.hpp"
#include "ffnn.hpp"


int32_t loadInt32(BinaryReader& reader)
{
    int32_t result{};
    char* buffer = reinterpret_cast<char*>(&result);
    reader.readInto(buffer[3]);
    reader.readInto(buffer[2]);
    reader.readInto(buffer[1]);
    reader.readInto(buffer[0]);
    return result;
}

std::vector<uint8_t> loadLabels(std::string const& filename)
{
    BinaryReader reader{filename};
    int32_t magic_number = loadInt32(reader);
    int32_t sample_count = loadInt32(reader);
    // Load labels
    std::vector<uint8_t> data;
    data.reserve(sample_count);
    for (int32_t i{sample_count}; i--;) {
        data.emplace_back(reader.read<uint8_t>());
    }
    return data;
}

std::vector<std::vector<uint8_t>> loadImages(std::string const& filename)
{
    BinaryReader reader{filename};
    int32_t magic_number = loadInt32(reader);
    int32_t sample_count = loadInt32(reader);
    int32_t row_count = loadInt32(reader);
    int32_t col_count = loadInt32(reader);
    // Load images
    std::vector<std::vector<uint8_t>> data;
    data.resize(sample_count);
    for (int32_t i{0}; i < sample_count; ++i) {
        data[i].resize(row_count * col_count);
        for (int32_t x{0}; x < col_count * row_count; ++x) {
            data[i][x] = reader.read<uint8_t>();
        }

    }
    return data;
}

int main()
{
    const uint32_t window_width  = 1600;
    const uint32_t window_height = 900;
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;

    auto const labels = loadLabels("mnist/labels");
    auto const images = loadImages("mnist/images");

    ffnn::FFNeuralNetwork network({784, 100, 10});
    ffnn::Optimizer optimizer(0.05f);

    std::vector<Eigen::VectorXf> samples;
    std::vector<Eigen::VectorXf> expected;

    auto const sample_count = labels.size();
    for (uint32_t i{0}; i < sample_count; ++i) {
        Eigen::VectorXf input;
        Eigen::VectorXf output;
        input.resize(784);
        output.resize(10);
        for (uint32_t pxl{0}; pxl < 784; ++pxl) {
            input[pxl] = to<float>(images[i][pxl]) / 255.0f;
        }

        uint8_t const exp = labels[i];
        for (uint32_t pxl{0}; pxl < 10; ++pxl) {
            output[pxl] = pxl == exp ? 1.0f : 0.0f;
        }

        samples.push_back(input);
        expected.push_back(output);
    }

    for (uint32_t i{0}; i < 1000; i++) {
        optimizer.train(network, samples, expected);
    }

    /*uint32_t image_idx = 1;
    sf::Image image;
    image.create(28, 28);
    uint32_t pxl{0};
    for (int32_t y{0}; y < 28; ++y) {
        for (int32_t x{0}; x < 28; ++x) {
            uint8_t const color = images[image_idx][pxl];
            image.setPixel(x, y, {color, color, color});
            ++pxl;
        }
    }
    std::cout << int(labels[image_idx]) << std::endl;
    sf::Texture texture;
    texture.loadFromImage(image);*/

    return 0;
}
