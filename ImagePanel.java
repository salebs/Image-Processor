import java.awt.event.*;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.awt.AWTException;
import java.awt.Graphics2D;
import java.awt.LayoutManager;
import java.awt.MouseInfo;
import java.awt.Point;
import java.awt.PointerInfo;
import java.awt.Robot;

/**
 * This class handles the information of the process' thread. Enables the user to see the progress of the prcoess and cancel the process if wanted.
 * 
 * @author Benjamin Sale
 */
class ImagePanel extends JFrame{

    JButton next;

    int panelWidth;
    int panelHeight;

    File[] images;
    BufferedImage image;
    int imageIndex;

    DragRect customRect;
    int rectWidth;
    int rectHeight;
    Double rectRatio;

    public void click(Point p) {
        try {
            PointerInfo pointerInfo = MouseInfo.getPointerInfo();
            Point mouseLocation = pointerInfo.getLocation();
            int oldX = (int) mouseLocation.getX();
            int oldY = (int) mouseLocation.getY();

            int newX = (int) p.getX() + Integer.valueOf(panelWidth/2);
            int newY = (int) p.getY() + Integer.valueOf(panelHeight/2);

            Robot robot = new Robot();
            robot.mouseMove(newX, newY);
            robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
            robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
            robot.mouseMove(oldX, oldY);
        } catch (AWTException e) {
            e.printStackTrace();
        }

    }

    public void resizeImage() {
        BufferedImage resizedImage = new BufferedImage(panelWidth, panelHeight, image.getType());
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(image, 0, 0, panelWidth, panelHeight, null);
        g2d.dispose();
        image = resizedImage;
    }

    public void getPatch() {
        int rectangleWidth = customRect.width;
        int rectangleHeight = customRect.height;
        int rectangleX = customRect.x;
        int rectangleY = customRect.y;
        if (rectangleWidth < 0 & rectangleHeight < 0) {
            rectangleWidth = Math.abs(rectangleWidth);
            rectangleHeight = Math.abs(rectangleHeight);
            rectangleX = rectangleX - rectangleWidth;
            rectangleY = rectangleY - rectangleHeight;
        } else if (rectangleWidth < 0) {
            rectangleWidth = Math.abs(rectangleWidth);
            rectangleX = rectangleX - rectangleWidth;
        } else if (rectangleHeight < 0) {
            rectangleHeight = Math.abs(rectangleHeight);
            rectangleY = rectangleY - rectangleHeight;
        }
        Double rectangleRatio = (double) Math.round((100 * (rectangleWidth * rectangleHeight) / (panelWidth * panelHeight)));
        rectWidth += rectangleWidth;
        rectHeight += rectangleHeight;
        rectRatio += rectangleRatio;
        BufferedImage patch = image.getSubimage(rectangleX, rectangleY, rectangleWidth, rectangleHeight);
        File outputfile = new File("positivePatches\\" + images[imageIndex].getName());
        try {
            ImageIO.write(patch, "jpg", outputfile);
        } catch (IOException e1) {
            System.out.println("Cannot save patch.");
            e1.printStackTrace();
        }
    }

    /**
     * Create a window for every time the process button is hit. Includes a way for the user to cancel the process.
     * @throws IOException
     */
    public ImagePanel(JButton button, String[] args) throws IOException {    
        setTitle("Image Panel");
        setSize(500, 500);
        setLayout((LayoutManager)null);
        setLocation(950, 100);

        rectWidth = 0;
        rectHeight = 0;
        rectRatio = (double) 0;
        File directory = new File("positive");
        images = directory.listFiles();
        imageIndex = 0;
        image = ImageIO.read(images[imageIndex]);
        panelWidth = 500;
        panelHeight = 250;
        resizeImage();
        customRect = new DragRect(image);
        customRect.setBounds(0, 0, panelWidth, panelHeight);

        String instructionTxt = "Drag and release on the image to draw a rectangle around the object.\n" +
                                "The rectangle can be moved around after being drawn if it is clicked \n" +
                                "within its region. Another rectangle replaces the previous rectangle\n" +
                                "if the image is clicked outside of the previous rectangle's region.\n" +
                                "Hit the button 'START' to see the first image and 'NEXT' to see the next\n" +
                                "image. When the last image is presented, hit 'ADVANCE' to run the program\n" +
                                "with your data.";
        JTextArea instructions = new JTextArea(instructionTxt);
        instructions.setEditable(false);
        instructions.setBounds(50, 50, 400, 200);
        add(instructions);

        next = new JButton("START");
        next.setActionCommand("start");
        next.addActionListener(new ActionListener() { 
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    if (imageIndex == images.length - 2) {
                        next.setText("ADVANCE");
                        next.setActionCommand("advance");
                    }
                    if (e.getActionCommand() == "start") {
                        next.setText("NEXT");
                        next.setActionCommand("next");
                        remove(instructions);
                        Point point = getLocation();
                        click(point);
                        add(customRect);
                    } else if (e.getActionCommand() == "next") { 
                        getPatch();
                        remove(customRect);
                        Point point = getLocation();
                        click(point);
                        imageIndex++;
                        image = ImageIO.read(images[imageIndex]);
                        resizeImage();
                        customRect = new DragRect(image);
                        customRect.setBounds(0, 0, panelWidth, panelHeight);
                        add(customRect);
                    } else if (e.getActionCommand() == "advance") {
                        getPatch();
                        dispose();
                        String command = "python " + "Process.py";
                        for (String arg : args) { command += " " + arg; }
                        float overallWidth = rectWidth / images.length;
                        float overallHeight = rectHeight / images.length;
                        Double overallRatio = rectRatio / images.length;
                        command += " " + overallWidth + " " + overallHeight + " " + overallRatio;
                        System.out.println(command);
                        ProcessBuilder processBuilder = new ProcessBuilder(command.split(" "));
                        Process process;
                        StatusWindow status;
                        try {
                            process = processBuilder.start();
                            status = new StatusWindow(process, button);
                            status.setVisible(true);
                        } catch (IOException e1) {
                            System.out.println("Cannot execute file.");
                            e1.printStackTrace();
                        }
                    }
                } catch (IOException e1) {
                    System.out.println("Cannot save patch.");
                    e1.printStackTrace();
                }
            }
        });
        next.setBounds(100, 350, 100, 25);
        add(next);

        JButton cancel = new JButton("CANCEL");
        cancel.setActionCommand("cancel");
        cancel.addActionListener(new ActionListener() { 
            public void actionPerformed(ActionEvent e) { 
                button.setEnabled(true);
            }
        });
        cancel.setBounds(300, 350, 100, 25);
        add(cancel);
    }
}
