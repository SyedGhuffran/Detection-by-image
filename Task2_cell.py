import cv2
import pytesseract
import webcolors

# Function to get the name of the color from RGB values
def get_color_name(rgb_tuple):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        closest_name = min((webcolors.CSS3_HEX_TO_NAMES[color], color) for color in webcolors.CSS3_HEX_TO_NAMES)[1]
    return closest_name

# Path to the image
image_path = 'IMG-20240524-WA0001.jpg'

# Load the image
image = cv2.imread(image_path)

if image is None:
    print("Image not found or unable to open")
else:
    # Convert the image to grayscale for OCR
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Configure parameters for tesseract
    custom_config = r'--oem 3 --psm 6'

    # Perform OCR on the image and get details as a dictionary
    details = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT, config=custom_config)

    # Number of boxes to consider
    n_boxes = len(details['text'])

    # Iterate through each detected text box
    for i in range(n_boxes):
        if int(details['conf'][i]) > 60:  # Consider only confident results
            # Extract the bounding box coordinates
            (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])

            # Optionally expand the bounding box slightly to better encompass the text
            cell_x = x - 10
            cell_y = y - 10
            cell_w = w + 20
            cell_h = h + 20

            # Draw the bounding box on the image
            cv2.rectangle(image, (cell_x, cell_y), (cell_x + cell_w, cell_y + cell_h), (0, 255, 0), 2)

            # Extract a small region of the cell to determine its color
            cell_region = image[y:y+h, x:x+w]
            avg_color_per_row = cv2.mean(cell_region)
            avg_color = (int(avg_color_per_row[0]), int(avg_color_per_row[1]), int(avg_color_per_row[2]))  # BGR format

            # Get color name
            color_name = get_color_name(avg_color)

            # Text to display
            cell_text = details['text'][i]
            text_size = cv2.getTextSize(cell_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]  # Adjusted size

            # Display text type, size, and color on the image
            text_display = f'Text: {cell_text} | Size: {w}x{h} | Color: {color_name} | Font: Simplex'
            cv2.putText(image, text_display, (cell_x, cell_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)  # Black color and smaller size

    # Display the image with bounding boxes and text info
    cv2.imshow('Detected Cells', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
