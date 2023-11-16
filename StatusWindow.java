import java.awt.LayoutManager;
import java.awt.event.*;
import javax.swing.*;
import java.io.IOException;

/**
 * This class handles the information of the process' thread. Enables the user to see the progress of the prcoess and cancel the process if wanted.
 * 
 * @author Benjamin Sale
 */
class StatusWindow extends JFrame{

    /**
     * Create a window for every time the process button is hit. Includes a way for the user to cancel the process.
     * @throws IOException
     */
    public StatusWindow(Process process, JButton button) throws IOException {    
        setTitle("Process Status");
        setSize(500, 500);
        setLayout((LayoutManager)null);
        setLocation(950, 100);

        add(new JLabel());

        JButton cancel = new JButton("CANCEL");
        cancel.setActionCommand("cancel");
        cancel.addActionListener(new ActionListener() { 
            public void actionPerformed(ActionEvent e) { 
                process.destroy();
                button.setEnabled(true);
                dispose();
            } 
        });
        cancel.setBounds(200, 400, 100, 25);
        add(cancel);
    }
}
