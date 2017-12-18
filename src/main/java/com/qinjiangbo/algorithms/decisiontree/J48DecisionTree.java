package com.qinjiangbo.algorithms.decisiontree;

import com.google.common.collect.Lists;
import com.google.common.io.Resources;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.Random;

/**
 * @date: 28/11/2017 10:33 AM
 * @author: qinjiangbo@github.io
 * @description:
 *      使用J48决策树对乳腺癌数据集进行训练和预测
 */
public class J48DecisionTree {

    /**
     * 训练数据集和测试数据集是相同的，用于后面交叉验证
     */
    private static final String TRAINING_DATASET_FILENAME = "decisiontree/breast-cancer.arff";

    public static void main(String[] args) {
        try {
            Instances instances = loadDataSet(TRAINING_DATASET_FILENAME);
            // 去除第一列[index=0]
            instances = filterAttributes(instances);
            // 8,10,10,8,7,10,9,7,1
            List<Integer> data = Lists.newArrayList(4,1,1,1,2,1,2,1,1);
            // 进行预测
            String classOfData = predict(data, instances);
            System.out.println("class of data is: " + classOfData);
            // 模型评价
            process();
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

    /**
     * 训练生成分类器
     * @return
     */
    public static Classifier generateClassifier() throws Exception{
        Instances instances = loadDataSet(TRAINING_DATASET_FILENAME);
        // 过滤后的数据集
        instances = filterAttributes(instances);
        // 初始化分类器
        Classifier j48 = new J48();
        // 训练该数据集
        j48.buildClassifier(instances);
        return j48;
    }

    /**
     * 过滤掉部分属性
     * @param instances 待过滤数据集
     * @return
     */
    public static Instances filterAttributes(Instances instances) {
        // 去掉第一个属性
        int[] indices = new int[]{0};
        // 子数据集
        Instances subset = null;
        // 属性移除过滤器
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indices);
        remove.setInvertSelection(false);
        try {
            remove.setInputFormat(instances);
            subset = Filter.useFilter(instances, remove);
        } catch (Exception e) {
            e.printStackTrace();
        }
        subset.setClassIndex(subset.numAttributes()-1);
        return subset;
    }

    /**
     * 打印出当前数据最可能所属的类别
     * @return
     */
    public static String predict(List<Integer> data, Instances trainingSet) throws Exception{
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
}
