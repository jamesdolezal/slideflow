import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.PathROIToolsAwt
import qupath.lib.scripting.QPEx

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.geom.AffineTransform
import java.awt.image.AffineTransformOp
import java.awt.image.DataBufferByte

def augmentation = true
def export = true
def extract_um = 280
def tile_px = 512

setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049 ", "Stain 2" : "Eosin", "Values 2" : "0.2159 0.8012 0.5581 ", "Background" : " 255 255 255 "}');
selectAnnotations();
runPlugin('qupath.lib.algorithms.TilerPlugin', String.format('{"tileSizeMicrons": %d,  "trimToROI": false,  "makeAnnotations": true,  "removeParentAnnotation": false}', extract_um));

def imageData = QPEx.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
def server = imageData.getServer()

def name = server.getShortServerName()
def home_dir = "/Users/james/thyroid/svs/tiles/" + name
QPEx.mkdirs(home_dir)
def path = buildFilePath(home_dir, String.format("Tile_coords_%s.txt", name))
def ann_path = buildFilePath(home_dir, String.format("%s.qptxt", name))
def tile_file = new File(path)
def ann_file = new File(ann_path)
tile_file.text = ''
ann_file.text = ''

for (obj in annotations) {
    if (obj.isAnnotation()) {
        def roi = obj.getROI()

        // Ignore empty annotations
        if (roi == null) {
            continue
        }
        // If rectangle, assume image tile, export
        if (roi.getClass() == qupath.lib.roi.RectangleROI) {
            def region = RegionRequest.createInstance(server.getPath(), 1.0, roi)
            String tile_name = String.format('%s_(%d,%d)',
                name,
                region.getX(),
                region.getY(),
            )
            def old_img = server.readBufferedImage(region)
            int width_old = old_img.getWidth()

            // Check if tile is mostly background
            // If >50% of pixels >240, then discard
            def gray_list = []
            for (int i=0; i < width_old; i++) {
                for (int j=0; j < width_old; j++) {
                    int gray = old_img.getRGB(i, j)& 0xFF;
                    gray_list << gray
                }
            }
            int median_px_i = (width_old * width_old) / 2
            median_px = gray_list.sort()[median_px_i]
            if (median_px > 220) { 
                print("Tile has >50% brightness >240, discarding")
                continue
            }
            // Write image tile coords to text file
            tile_file << roi.getPolygonPoints() << System.lineSeparator()
            BufferedImage img = new BufferedImage(tile_px, tile_px, old_img.getType())
            if (export) {
                // Resize tile
                AffineTransform resize = new AffineTransform()
                resize_factor = tile_px / width_old
                resize.scale(resize_factor, resize_factor)
                AffineTransformOp resizeOp = new AffineTransformOp(resize, AffineTransformOp.TYPE_BILINEAR)
                resizeOp.filter(old_img, img)
                w = img.getWidth()
                h = img.getHeight()
    
                def fileImage = new File(home_dir, tile_name + ".jpg")
                print("Writing image tiles for tile " + tile_name)
                ImageIO.write(img, "JPG", fileImage)
            }

            if (augmentation && export) {
                AffineTransform rotateTransform = new AffineTransform()
                rotateTransform.translate(h/2, w/2)
                rotateTransform.rotate(Math.PI/2)
                rotateTransform.translate(-w/2, -h/2)
                AffineTransformOp rotateOp = new AffineTransformOp(rotateTransform, AffineTransformOp.TYPE_BILINEAR)

                AffineTransform vertTransform = AffineTransform.getScaleInstance(1, -1)
                vertTransform.translate(0, -h)
                AffineTransformOp vertOp = new AffineTransformOp(vertTransform, AffineTransformOp.TYPE_BILINEAR)

                AffineTransform horzTransform = AffineTransform.getScaleInstance(-1, 1)
                horzTransform.translate(-w, 0)
                AffineTransformOp horzOp = new AffineTransformOp(horzTransform, AffineTransformOp.TYPE_BILINEAR)
                
                def aug1file = new File(home_dir, tile_name + "_aug1.jpg")
                def aug2file = new File(home_dir, tile_name + "_aug2.jpg")
                def aug3file = new File(home_dir, tile_name + "_aug3.jpg")
                def aug4file = new File(home_dir, tile_name + "_aug4.jpg")
                def aug5file = new File(home_dir, tile_name + "_aug5.jpg")
                def aug6file = new File(home_dir, tile_name + "_aug6.jpg")
                def aug7file = new File(home_dir, tile_name + "_aug7.jpg")

                BufferedImage aug1 = new BufferedImage(h, w, img.getType())
                BufferedImage aug2 = new BufferedImage(h, w, img.getType())
                BufferedImage aug3 = new BufferedImage(h, w, img.getType())
                BufferedImage aug4 = new BufferedImage(h, w, img.getType())
                BufferedImage aug5 = new BufferedImage(h, w, img.getType())
                BufferedImage aug6 = new BufferedImage(h, w, img.getType())
                BufferedImage aug7 = new BufferedImage(h, w, img.getType())

                rotateOp.filter(img, aug1)
                horzOp.filter(img, aug2)
                horzOp.filter(aug1, aug3)
                vertOp.filter(img, aug4)
                vertOp.filter(aug1, aug5)
                vertOp.filter(aug2, aug6)
                vertOp.filter(aug3, aug7)

                ImageIO.write(aug1, "JPG", aug1file)
                ImageIO.write(aug2, "JPG", aug2file)
                ImageIO.write(aug3, "JPG", aug3file)
                ImageIO.write(aug4, "JPG", aug4file)
                ImageIO.write(aug5, "JPG", aug5file)
                ImageIO.write(aug6, "JPG", aug6file)
                ImageIO.write(aug7, "JPG", aug7file)
            }
        } else {
            print("Name: " + obj.name)
            points = roi.getPolygonPoints()
            ann_file << "Name: " + obj.name << System.lineSeparator()
            for (point in points) {
                p_x = point.getX()
                p_y = point.getY()
                point_string = p_x + ", " + p_y
                ann_file << point_string << System.lineSeparator()
            }
            ann_file << "end" << System.lineSeparator()
        }
    }
}
print("Finished!")