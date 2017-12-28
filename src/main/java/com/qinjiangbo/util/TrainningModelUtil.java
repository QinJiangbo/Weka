package com.qinjiangbo.util;

import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

/**
 * @date: 28/12/2017 10:34 AM
 * @author: qinjiangbo@github.io
 * @description: 主要是用来保存训练好的模型到文件，以及从文件中读取训练好的模型
 */
public class TrainningModelUtil {

    /**
     * 模型保存的目录
     */
    private static final String MODEL_STORAGE_DIR = "/Users/richard/Documents/Weka Models/";
    /**
     * 模型文件的后缀
     */
    private static final String MODEL_EXTENSION = ".model";

    /**
     * 保存模型名称
     * @param classifier
     * @param modelName 模型名称
     */
    public static void saveModel(Classifier classifier, String modelName) {
        try {
            SerializationHelper.write(MODEL_STORAGE_DIR +
                    modelName + MODEL_EXTENSION, classifier);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 读取模型
     * @param modelName
     * @param <T>
     * @return
     */
    public static <T> T readModel(String modelName) {
        Classifier classifier = null;
        try {
            classifier = (Classifier) SerializationHelper.read(MODEL_STORAGE_DIR +
                    modelName + MODEL_EXTENSION);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return (T) classifier;
    }
}
