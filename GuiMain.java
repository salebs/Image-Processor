import java.awt.GridLayout;
import java.awt.event.*;
import javax.swing.*;

import java.io.IOException;

/**
 * This class enables the user to easily access the Image Processor with specified parameters and without touching any code.
 * 
 * @author Benjamin Sale
 */
public class GuiMain extends JFrame implements ActionListener{

    /**
     * The fields will have the user-defined arguments for the terminal command.
     */
    private static JTextField[] fields;

    /**
     * The view will have the flags to determine if the user wants to see the trained or testing images.
     */
    private static JCheckBox[] view;

    /**
     * The button is the process button.
     */
    private JButton[] buttons;

    /**
     * When the process button is hit, a process thread is spun off to construct an Image Processor with the user specified paramters. It will also disable the process button and create a GuiWindow.
     * 
     * @param args a string array with the appropriate terminal command arguments
     */
    public void run(String[] args) {
        try {
            ImagePanel image = new ImagePanel(buttons[0], args);
            image.setVisible(true);
        } catch (IOException e) {
            System.out.println("Exception is caught");
            e.printStackTrace();
        }
    }

    public void predict() {
        String command = "python " + "Predict.py";
        System.out.println(command);
        ProcessBuilder processBuilder = new ProcessBuilder(command.split(" "));
        Process process;
        StatusWindow status;
        try {
            process = processBuilder.start();
            status = new StatusWindow(process, buttons[2]);
            status.setVisible(true);
        } catch (IOException e1) {
            System.out.println("Cannot execute file.");
            e1.printStackTrace();
        }
    }

    public void adjust(String[] args) {
        String command = "python " + "Adjust.py";
        for (String arg : args) { command += " " + arg; }
        System.out.println(command);
        ProcessBuilder processBuilder = new ProcessBuilder(command.split(" "));
        Process process;
        StatusWindow status;
        try {
            process = processBuilder.start();
            status = new StatusWindow(process, buttons[1]);
            status.setVisible(true);
        } catch (IOException e1) {
            System.out.println("Cannot execute file.");
            e1.printStackTrace();
        }
    }


    /**
     * Ensure all fields are filled out in GUI and starts the process based off the user-defined parameters.
     */
    public void actionPerformed(ActionEvent event) {
        System.out.println("clicked");
        if (event.getActionCommand() == "model" | event.getActionCommand() == "adjust") {
            String[] args = new String[14];
            for (int i=0; i<fields.length; i++) {
                int j = i;
                String fieldText = fields[j].getText();
                System.out.println(fieldText);
                if (fieldText == "") { 
                    JOptionPane.showMessageDialog(null, "Enter information for all fields.", "Warning", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                args[j] = fieldText; 
                
            }
            for (int i=0; i<view.length; i++) {
                boolean check = view[i].isSelected();
                String boxString;
                if (check) { boxString = "true"; }
                else { boxString = "false"; }
                args[fields.length + i] = boxString;
            }
            for (String arg : args) { System.out.println(arg); }
            if (event.getActionCommand() == "adjust") { adjust(args); }
            else { run(args); }
        } else if (event.getActionCommand() == "predict") {
            predict();
        }
    }
    
    /**
     * Create a GUI and install all appropriate fields and check boxes to gather terminal command arguments, as well as process button.
     */
    public GuiMain() {
        fields = new JTextField[12];
        view = new JCheckBox[2];
        buttons = new JButton[3];

        setTitle("Image Processor");
        setLayout(new GridLayout(12, 3));
        setLocation(250, 100);


        JTextField modelTypeResponse = new JTextField("KNeighborsClassifier");
        modelTypeResponse.setName("model type");
        fields[0] = modelTypeResponse;
        JLabel modelType = new JLabel("Enter model type:");
        add(modelType);
        String[] items = {"KNeighborsClassifier", "MLPClassifier", "GridSearchCV"};
        JComboBox<String> comboBox = new JComboBox<String>(items);
        comboBox.setSelectedIndex(0);
        comboBox.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String selectedOption = (String) comboBox.getSelectedItem();
                modelTypeResponse.setText(selectedOption);
            }
        });
        add(comboBox);

        add(new JLabel());

        JLabel negativePatchCount = new JLabel("Enter negative amount:");
        add(negativePatchCount);
        JTextField negativePatchCountResponse = new JTextField("1");
        negativePatchCountResponse.setName("negative patch number");
        add(negativePatchCountResponse);
        fields[1] = negativePatchCountResponse;

        add(new JLabel());

        JLabel imageSize = new JLabel("Enter image size:");
        add(imageSize);
        JTextField imageSizeMinResponse = new JTextField("1000");
        imageSizeMinResponse.setName("image size minimum");
        add(imageSizeMinResponse);
        fields[2] = imageSizeMinResponse;
        JTextField imageSizeMaxResponse = new JTextField("500");
        imageSizeMaxResponse.setName("image size maximum");
        add(imageSizeMaxResponse);
        fields[3] = imageSizeMaxResponse;

        JLabel cellSize = new JLabel("Enter cell size:");
        cellSize.setBounds(20, 150, 150, 20);
        add(cellSize);
        JTextField cellSizeMinResponse = new JTextField("10");
        cellSizeMinResponse.setName("cell size min");
        add(cellSizeMinResponse);
        fields[4] = cellSizeMinResponse;
        JTextField cellSizeMaxResponse = new JTextField("10");
        cellSizeMaxResponse.setName("cell size min");
        add(cellSizeMaxResponse);
        fields[5] = cellSizeMaxResponse;

        JLabel blockSize = new JLabel("Enter block size:");
        add(blockSize);
        JTextField blockSizeMinResponse = new JTextField("1");
        blockSizeMinResponse.setName("block size minimum");
        add(blockSizeMinResponse);
        fields[6] = blockSizeMinResponse;
        JTextField blockSizeMaxResponse = new JTextField("1");
        blockSizeMaxResponse.setName("block size maximum");
        add(blockSizeMaxResponse);
        fields[7] = blockSizeMaxResponse;

        JLabel stepSize = new JLabel("Enter step size:");
        add(stepSize);
        JTextField stepSizeResponse = new JTextField("0.5");
        stepSizeResponse.setName("step size");
        add(stepSizeResponse);
        fields[8] = stepSizeResponse;

        add(new JLabel());

        JLabel scales = new JLabel("Enter scales:");
        add(scales);
        JTextField scalesMinResponse = new JTextField("0.5");
        scalesMinResponse.setName("scales minimum");
        add(scalesMinResponse);
        fields[9] = scalesMinResponse;
        JTextField scalesMaxResponse = new JTextField("2");
        scalesMaxResponse.setName("scales maximum");
        add(scalesMaxResponse);
        fields[10] = scalesMaxResponse;

        JLabel neighbor = new JLabel("Enter neighbor count:");
        add(neighbor);
        JTextField neighborResponse = new JTextField("3");
        neighborResponse.setName("neighbor count");
        add(neighborResponse);
        fields[11] = neighborResponse;

        add(new JLabel());
        add(new JLabel());

        JCheckBox displayTrain = new JCheckBox("display training images"); 
        add(displayTrain);
        view[0] = displayTrain;

        JCheckBox displayTest = new JCheckBox("display testing images"); 
        add(displayTest);
        view[1] = displayTest;

        JButton process = new JButton("MODEL");
        process.setActionCommand("model");
        process.addActionListener(this);
        add(process);
        buttons[0] = process;

        JButton adjust = new JButton("ADJUST");
        adjust.setActionCommand("adjust");
        adjust.addActionListener(this);
        add(adjust);
        buttons[1] = adjust;

        JButton predict = new JButton("PREDICT");
        predict.setActionCommand("predict");
        predict.addActionListener(this);
        add(predict);
        buttons[2] = predict;

        pack();
    }

    public static void main(String[] args) {
        GuiMain window = new GuiMain();
        window.setDefaultCloseOperation(3);
        window.setVisible(true);
     }
}
