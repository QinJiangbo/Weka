package com.qinjiangbo.algorithms.decisiontree;

import com.google.common.collect.Lists;
import com.google.common.io.Resources;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.Random;

/**
 * @date: 03/12/2017 9:52 PM
 * @author: qinjiangbo@github.io
 * @description:
 *      本类主要用于学习《机器学习》西瓜数据集的处理和预测
 */
public class WatermelonClassifier {
    /**
     * 训练数据集和测试数据集是相同的，用于后面交叉验证
     */
    private static final String TRAINING_DATASET_FILENAME = "decisiontree/watermelon-training.arff";
    private static final String TESTING_DATASET_FILENAME = "decisiontree/watermelon-test.arff";

    public static void main(String[] args) {
        try {
            Instances instances = loadDataSet(TRAINING_DATASET_FILENAME);
            // 青绿,蜷缩,沉闷,清晰,凹陷,硬滑 [是]
            // 浅白,蜷缩,浊响,模糊,平坦,软粘 [否]
            List<String> data =
                    Lists.newArrayList("浅白","蜷缩","浊响","模糊","平坦","软粘");
            // 进行预测
            String classOfData = predict(data, instances);
            System.out.println("class of data is: " + classOfData);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 加载数据集
     * @param fileName 训练集文件地址
     * @return
     */
    public static Instances loadDataSet(String fileName) {
        Instances instances = null;
        try {
            URL url = Resources.getResource(fileName);
            File file = new File(url.getPath());
            ArffLoader arffLoader = new ArffLoader();
            arffLoader.setFile(file);
            instances = arffLoader.getDataSet();
        } catch (IOException e) {
            e.printStackTrace();
        }
        instances.setClassIndex(instances.numAttributes()-1);
        return instances;
    }

    /**
     * 训练生成分类器
     * @return
     */
    public static Classifier generateClassifier() throws Exception{
        Instances instances = loadDataSet(TRAINING_DATASET_FILENAME);
        // 初始化分类器
        Classifier j48 = new J48();
        // 训练该数据集
        j48.buildClassifier(instances);
        return j48;
    }

    /**
     * 打印出当前数据最可能所属的类别
     * @return
     */
    public static String predict(List<String> data, Instances trainingSet) throws Exception{
        Classifier j48 = generateClassifier();
        // 创建Instance
        Instance instance = new DenseInstance(trainingSet.numAttributes());
        // 分别添加待预测特征值
        for (int i = 0; i < data.size(); i++) {
            instance.setValue(trainingSet.attribute(i), data.get(i));
        }
        // 需要能访问数据集
        instance.setDataset(trainingSet);
        // 得出最可能所属类别
        int index = (int)j48.classifyInstance(instance);
        return trainingSet.classAttribute().value(index);
    }

    /**
     * 操作数据集
     * @throws Exception
     */
    public static void process() throws Exception{
        Instances instances = loadDataSet(TRAINING_DATASET_FILENAME);
        Classifier j48 = generateClassifier();
        // 使用同一个数据集进行交叉验证
        Evaluation evaluation = new Evaluation(instances);
        evaluation.crossValidateModel(j48, instances, 10, new Random(1));

        /** 打印算法的汇总信息 */
        System.out.println("** Decision Tress Evaluation with Datasets **");
        System.out.println(evaluation.toSummaryString());
        System.out.print(" the expression for the input data as per algorithm is ");
        System.out.println(j48);
        System.out.println(evaluation.toMatrixString());
        System.out.println(evaluation.toClassDetailsString());
    }
}
