import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;

import javax.swing.JPanel;
import java.awt.Rectangle;

class DragRect extends JPanel {

    private final class MouseDrag extends MouseAdapter {
        private boolean dragging = false;
        private boolean expanding = false;
        private Point last;
        private ResizeDirection resizeDirection;
        private Rectangle rectangle;

        private enum ResizeDirection {
            NONE, N, NE, E, SE, S, SW, W, NW
        }

        public void resize(ResizeDirection direction, Point mousePosition) {
            switch (direction) {
                case N:
                    resizeNorth(mousePosition);
                    break;
                case NE:
                    resizeNorth(mousePosition);
                    resizeEast(mousePosition);
                    break;
                case E:
                    resizeEast(mousePosition);
                    break;
                case SE:
                    resizeSouth(mousePosition);
                    resizeEast(mousePosition);
                    break;
                case S:
                    resizeSouth(mousePosition);
                    break;
                case SW:
                    resizeSouth(mousePosition);
                    resizeWest(mousePosition);
                    break;
                case W:
                    resizeWest(mousePosition);
                    break;
                case NW:
                    resizeNorth(mousePosition);
                    resizeWest(mousePosition);
                    break;
                default:
                    break;
            }
        }

        private void resizeNorth(Point mousePosition) {
            int deltaY = mousePosition.y - y;
            height -= deltaY;
            y = mousePosition.y;
        }

        private void resizeEast(Point mousePosition) {
            int deltaX = mousePosition.x - (x + width);
            width += deltaX;
        }

        private void resizeSouth(Point mousePosition) {
            int deltaY = mousePosition.y - (y + height);
            height += deltaY;
        }

        private void resizeWest(Point mousePosition) {
            int deltaX = mousePosition.x - x;
            width -= deltaX;
            x = mousePosition.x;
        }

        private ResizeDirection getResizeDirection(Point point) {
            int resizeMargin = lineWidth;
            int x = point.x;
            int y = point.y;
    
            if (y <= rectangle.y + resizeMargin) {
                if (x <= rectangle.x + resizeMargin) {
                    return ResizeDirection.NW;
                } else if (x >= rectangle.x + rectangle.width - resizeMargin) {
                    return ResizeDirection.NE;
                } else {
                    return ResizeDirection.N;
                }
            } else if (y >= rectangle.y + rectangle.height - resizeMargin) {
                if (x <= rectangle.x + resizeMargin) {
                    return ResizeDirection.SW;
                } else if (x >= rectangle.x + rectangle.width - resizeMargin) {
                    return ResizeDirection.SE;
                } else {
                    return ResizeDirection.S;
                }
            } else {
                if (x <= rectangle.x + resizeMargin) {
                    return ResizeDirection.W;
                } else if (x >= rectangle.x + rectangle.width - resizeMargin) {
                    return ResizeDirection.E;
                } else {
                    return ResizeDirection.NONE;
                }
            }
        }

        private Rectangle getRectangle() {
            int newX = x;
            int newY = y;
            if (width < 0) { newX = x + width; }
            if (height < 0) { newY = y + height; }
            return new Rectangle(newX, newY, Math.abs(width), Math.abs(height));
        }

        @Override
        public void mousePressed(MouseEvent m) {
            last = m.getPoint();
            expanding = isOnEdge(last);
            dragging = isInsideRect(last);
            if (!dragging && !expanding) {
                x = last.x;
                y = last.y;
                width = 0;
                height = 0;
            }
            repaint();
        }

        @Override
        public void mouseReleased(MouseEvent m) {
            last = null;
            dragging = false;
            expanding = false;
            repaint();
        }

        @Override
        public void mouseDragged(MouseEvent m) {
            int dx = m.getX() - last.x;
            int dy = m.getY() - last.y;
            if (dragging) {
                x += dx;
                y += dy;
            } else if (expanding) {
                Point p = new Point(m.getX(), m.getY());
                resizeDirection = getResizeDirection(p);
                resize(resizeDirection, p);
            } else {
                width += dx;
                height += dy;
            }
            last = m.getPoint();
            rectangle = getRectangle();
            repaint();
        }
    }

    public int x;
    public int y;
    public int width;
    public int height;

    private int lineWidth;

    private MouseDrag mouseDrag;

    private BufferedImage image;

    public DragRect(BufferedImage image) {
        this.image = image;
        lineWidth = 6;
        mouseDrag = new MouseDrag();
        addMouseListener(mouseDrag);
        addMouseMotionListener(mouseDrag);
    }

    public boolean isOnEdge(Point point) {
        int newX = x;
        int newY = y;
        if (width < 0) { newX = x + width; }
        if (height < 0) { newY = y + height; }
        Rectangle validRectangle = new Rectangle(newX - lineWidth, newY - lineWidth, Math.abs(width) + 2*lineWidth, Math.abs(height) + 2*lineWidth); 
        Rectangle invalidRectangle = new Rectangle(newX, newY, Math.abs(width), Math.abs(height));
        return validRectangle.contains(point) && !(invalidRectangle.contains(point));
    }

    public boolean isInsideRect(Point point) {
        int newX = x;
        int newY = y;
        if (width < 0) { newX = x + width; }
        if (height < 0) { newY = y + height; }
        return new Rectangle2D.Float(newX, newY, Math.abs(width), Math.abs(height)).contains(point);
    }

    public void drawThickerRectangle(Graphics2D g2d) {
        g2d.setStroke(new BasicStroke(lineWidth));
        g2d.drawRect(x, y, width, height);
    }
    
    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(image, 0, 0, this);
        g.setColor(Color.RED);
        drawThickerRectangle((Graphics2D) g);

        int newX = x;
        int newY = y;
        if (width < 0) { newX = x + width; }
        if (height < 0) { newY = y + height; }
        g.drawRect(newX, newY, Math.abs(width), Math.abs(height));
    }
}
