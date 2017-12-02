package com.qinjiangbo.algorithms.decisiontree;

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
            Classifier j48 = generateClassifier();
            // 8,10,10,8,7,10,9,7,1
            Attribute attr01 = instances.attribute("clump_thickness");
            Attribute attr02 = instances.attribute("uniformity_of_cell_size");
            Attribute attr03 = instances.attribute("uniformity_of_cell_shape");
            Attribute attr04 = instances.attribute("marginal_adhesion");
            Attribute attr05 = instances.attribute("single_epithelial_cell_size");
            Attribute attr06 = instances.attribute("bare_nuclei");
            Attribute attr07 = instances.attribute("bland_chromatin");
            Attribute attr08 = instances.attribute("normal_nucleoli");
            Attribute attr09 = instances.attribute("mitoses");

            // 创建Instance
            Instance instance = new DenseInstance(instances.numAttributes());
            instance.setValue(attr01, 8);
            instance.setValue(attr02, 10);
            instance.setValue(attr03, 10);
            instance.setValue(attr04, 8);
            instance.setValue(attr05, 7);
            instance.setValue(attr06, 10);
            instance.setValue(attr07, 9);
            instance.setValue(attr08, 7);
            instance.setValue(attr09, 1);
            instance.setDataset(instances);
            double[] distribution = j48.distributionForInstance(instance);

            for (double dist : distribution) {
                System.out.println(dist);
            }

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
        // 去掉第一个属性
        int[] indices = new int[]{0};
        // 过滤后的数据集
        instances = filterAttributes(instances, indices);
        // 初始化分类器
        Classifier j48 = new J48();
        // 训练该数据集
        j48.buildClassifier(instances);
        return j48;
    }

    /**
     * 过滤掉部分属性
     * @param instances 待过滤数据集
     * @param indices   需要移除的属性下标，下标从0开始
     * @return
     */
    public static Instances filterAttributes(Instances instances, int[] indices) {
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
}
