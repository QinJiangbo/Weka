package com.qinjiangbo.util;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.IOException;

/**
 * @date: 27/11/2017 5:06 PM
 * @author: qinjiangbo@github.io
 * @description:
 */
public class ArffCreatorUtil {

    /**
     * 将数据集保存为Arff文件
     * @param dataSet
     * @param arffName
     */
    public void saveDataToArff(Instances dataSet, String arffName) throws IOException {
        ArffSaver saver = new ArffSaver();
        saver.setFile(new File(arffName));
        saver.setDestination(new File(arffName));
        // 批量刷到磁盘
        saver.writeBatch();
    }
}
