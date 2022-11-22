import os
# Import the pdf library
from fpdf import FPDF


RESULTS_DIR = './results'
annotated = True

def generate_pdf(dataset, filenames):
    '''
    dataset: string, name of the pdf file
    filenames: list of strings, filepaths to the images to include in the pdf
    '''
    # Sort filenames if annotated
    if annotated:
        filenames.sort()
        # Reshape 1x15 list to 5x3 list
        filenames = [filenames[i:i+3] for i in range(0, len(filenames), 3)]
        # Transpose the list
        filenames = list(map(list, zip(*filenames)))
        # Flatten the list
        filenames = [item for sublist in filenames for item in sublist]


    # Make every 2nd image
    # Create a PDF with one page in landscape mode
    # Add 15 images in a 3x5 grid on the page
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)

    # Add the title
    pdf.cell(0, 10, txt='Plots for {}'.format(dataset), ln=1, align='C')

    # Add the images
    for i, filename in enumerate(filenames):
        # Add the image
        pdf.image(os.path.join(RESULTS_DIR, filename), x=(i % 5) * 60, y= 30 + (i // 5) * 60, w=60)
        # Add the filename in small text at top of image only if annotated
        if annotated:
            # Add filename as substring by removing the dataset name and the extension from front
            if 'csv' in filename:
                filename = filename[len(dataset) + 5:]
            else:
                filename = filename[len(dataset) + 6:]
            pdf.set_font('Arial', 'B', 10)
            #pdf.text(x=(i % 5) * 60, y= 30 + (i // 5) * 60, txt=filename)
           
            pdf.text(x=(i % 5) * 60, y= 30 + (i // 5) * 60, txt=str(i+1))
            
    
    # Save the pdf to "pdfs" directory
    #pdf.output(os.path.join('pdfs' + ('_annotated' if annotated else ''), '{}.pdf'.format(dataset)))
    pdf.output(os.path.join('pdfs' + ('_userstudy_numbered' if annotated else ''), '{}.pdf'.format(dataset)))



def main():
    # Get all filenames in the results directory
    filenames = os.listdir(RESULTS_DIR)

    dataset_to_files = {}

    for filename in filenames:
        dataset = filename.split('.')[0]
        if dataset not in dataset_to_files:
            dataset_to_files[dataset] = []
        dataset_to_files[dataset].append(filename)

    # Generate a PDF for each dataset
    for dataset, filenames in dataset_to_files.items():
        generate_pdf(dataset, filenames)

if __name__ == "__main__":
    main()