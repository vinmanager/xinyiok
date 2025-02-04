#include "onnxruntime_cxx_api.h"



int main() {

    // 1. 初始化环境

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    Ort::SessionOptions session_options;

    // 2. 创建会话

    Ort::Session session(env, "resnet18.onnx", session_options);



    // 3. 准备输入

    const char* input_names[] = {"input"};  // 与模型输入名一致

    float input_data[3*224*224] = {0};    // 实际数据需做归一化等预处理，此处默认先给0

    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(

        memory_info, input_data, 3*224*224, input_shape.data(), input_shape.size()

    );



    // 4. 执行推理

    const char* output_names[] = {"output"};

    auto output_tensors = session.Run(

        Ort::RunOptions{nullptr}, 

        input_names, &input_tensor, 1, 

        output_names, 1

    );



    // 5. 解析输出

    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    // 后处理（如argmax获取分类结果）

}