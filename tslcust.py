import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import warnings
import calendar
import datetime
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tslearn.utils import to_time_series
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def add_text_to_pdf(pdf_pages, text):
    """
    Add text to the PDF.

    Parameters:
        pdf_pages (PdfPages):
            PdfPages object to add text to.
        text (str):
            Text to be added to the PDF.
    """
    # Create a new figure for the text
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
    ax.axis('off')  # Hide axes

    # Save the figure to the PDF
    pdf_pages.savefig(fig)
    plt.close(fig)  # Close the figure to release memory

def plot_rainy_vs_non_rainy_days_bar(df,pdf):
    """
    Plot a bar chart of rainy vs non-rainy days for the entire DataFrame.

    Parameters:
        df (DataFrame):
            DataFrame containing the rainfall information.
    """
    # Calculate percentage of rainy days
    rainy_days_percentage = (df['BOM_Rainfallmm'] > 0).mean() * 100
    non_rainy_days_percentage = 100 - rainy_days_percentage

    # Plot bar chart
    plt.figure(figsize=(6, 6))
    plt.bar(['Rainy Days', 'Non-Rainy Days'], [rainy_days_percentage, non_rainy_days_percentage], color=['skyblue', 'orange'])
    plt.title('Rainy vs Non-Rainy Days')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)

    # Annotate exact percentage numbers on the chart
    for i, percentage in enumerate([rainy_days_percentage, non_rainy_days_percentage]):
        plt.text(i, percentage, f'{percentage:.2f}%', ha='center', va='bottom')

    pdf.savefig()
    plt.close()


def visualize_speed_distribution(df,pdf):
    """
    Visualize the distribution of the 'SpeedKmh' column in the DataFrame.

    Parameters:
        df (DataFrame):
            DataFrame containing the 'SpeedKmh' column.
    """
    # Create subplots
    fig, axs = plt.subplots(3, figsize=(10, 15))

    # Box plot
    sns.boxplot(df['SpeedKmh'], ax=axs[0])
    axs[0].set_title('Box Plot')

    # Histogram
    sns.histplot(df['SpeedKmh'], bins=20, kde=False, ax=axs[1])
    axs[1].set_title('Histogram')

    # Density plot
    sns.kdeplot(df['SpeedKmh'], ax=axs[2])
    axs[2].set_title('Density Plot')

    plt.tight_layout()

    pdf.savefig()
    plt.close()

def apply_kmeans(X, metric='euclidean', n_clusters=5, random_state=721):
    """
    Apply KMeans clustering on time series data.

    Parameters:
        X (array-like, shape=(n_samples, n_timestamps, n_features)):
            Time series data to cluster.
        metric (str, optional):
            The distance metric to use for clustering. Default is 'euclidean'.
            Other options include 'dtw' (dynamic time warping), 'softdtw' (soft dynamic time warping),
            'sbd' (squared binary divergence), 'kld' (Kullback-Leibler divergence), 'dtw_sakoechiba' 
            (DTW with Sakoe-Chiba constraint), 'dtw_itakura' (DTW with Itakura constraint).
        n_clusters (int, optional):
            The number of clusters to form. Default is 8.
        random_state (int, RandomState instance or None, optional):
            Determines random number generation for centroid initialization. 
            Pass an int for reproducible results across multiple function calls. Default is None.

    Returns:
        labels (array-like, shape=(n_samples,)):
            Cluster labels for each time series.
    """
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return labels

def compute_day_of_week_percentage(df):
    """
    Compute the percentage of each day of the week in each cluster.

    Parameters:
        df (DataFrame):
            DataFrame containing the cluster labels and datetime information.

    Returns:
        day_of_week_percentages (DataFrame):
            DataFrame containing the percentage of each day of the week in each cluster.
    """
    # Count of each day of the week in each cluster
    day_of_week_counts = df.groupby(['cluster_label', df['LocalDateTime'].dt.dayofweek]).size()

    # Total count of data points in each cluster
    total_counts = df['cluster_label'].value_counts()

    # Percentage of each day of the week in each cluster
    day_of_week_percentages = (day_of_week_counts / total_counts).unstack(fill_value=0) * 100

    # Convert day of week index to actual day names
    day_of_week_percentages.columns = [calendar.day_name[i] for i in range(7)]

    return day_of_week_percentages

def plot_cluster_day_of_week_bar(df, day_of_week_percentages,pdf):
    """
    Plot bar charts for each cluster, where each bar represents the percentages of the day of the week.

    Parameters:
        df (DataFrame):
            DataFrame containing the cluster labels and datetime information.
        day_of_week_percentages (DataFrame):
            DataFrame containing the percentage of each day of the week in each cluster.
    """
    # Determine the number of clusters
    n_clusters = df['cluster_label'].nunique()

    # Create subplots for each cluster
    fig, axs = plt.subplots(n_clusters, figsize=(10, 6 * n_clusters))

    # Plot bar chart for each cluster
    for cluster_label, ax in zip(range(n_clusters), axs):
        # Get percentages for the cluster
        percentages = day_of_week_percentages.loc[cluster_label].sort_values(ascending=False)

        # Plot bar chart
        ax.bar(percentages.index, percentages.values, color='skyblue')
        ax.set_title(f'Cluster {cluster_label + 1}')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Percentage')
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(calendar.day_abbr)))
        ax.set_xticklabels(calendar.day_abbr)

    plt.tight_layout()

    pdf.savefig()
    plt.close()

def plot_cluster_volume_time(df,pdf):
    """
    Plot volume on the y-axis and time of the day on the x-axis for each cluster.

    Parameters:
        df (DataFrame):
            DataFrame containing the cluster labels and datetime information.
    """
    # Determine the number of clusters
    n_clusters = df['cluster_label'].nunique()

    # Create subplots for each cluster
    fig, axs = plt.subplots(n_clusters, figsize=(10, 6 * n_clusters), sharex=True)

    # Plot volume vs. time for each cluster
    for cluster_label, ax in zip(range(n_clusters), axs):
        cluster_df = df[df['cluster_label'] == cluster_label]

        # Extract hour and minute from LocalDateTime
        cluster_df['Hour'] = cluster_df['LocalDateTime'].dt.hour
        cluster_df['Minute'] = cluster_df['LocalDateTime'].dt.minute

        # Filter data to include only the time range 00:00 to 23:45
        cluster_df = cluster_df[((cluster_df['Hour'] == 0) & (cluster_df['Minute'] >= 0)) | 
                        ((cluster_df['Hour'] > 0) & (cluster_df['Hour'] < 23)) |
                        ((cluster_df['Hour'] == 23) & (cluster_df['Minute'] <= 45))]

        # Plot volume vs. time
        ax.plot(cluster_df['Hour'] + cluster_df['Minute'] / 60, cluster_df['VolumeCount'])
        ax.set_title(f'Cluster {cluster_label + 1}')
        ax.set_xlabel('Time of Day (hours)')
        ax.set_ylabel('Volume')
        ax.set_xlim(0, 24)  # Set x-axis limit from 0 to 24 hours
        ax.grid(True)

    plt.tight_layout()

    pdf.savefig()
    plt.close()

def plot_cluster_volume_time2(df, pdf):
    """
    Plot volume on the y-axis and time of the day on the x-axis for each cluster.

    Parameters:
        df (DataFrame):
            DataFrame containing the cluster labels and datetime information.
        pdf (PdfPages):
            PdfPages object to save the plots to PDF.
    """
    # Determine the number of clusters
    n_clusters = df['cluster_label'].nunique()

    # Create subplots for each cluster
    fig, axs = plt.subplots(n_clusters, figsize=(10, 6 * n_clusters), sharex=True)

    # Plot volume vs. time for each cluster
    for cluster_label, ax in zip(range(n_clusters), axs):
        cluster_df = df[df['cluster_label'] == cluster_label]

        # Extract hour and minute from LocalDateTime
        cluster_df['Hour'] = cluster_df['LocalDateTime'].dt.hour
        cluster_df['Minute'] = cluster_df['LocalDateTime'].dt.minute

        # Filter data to include only the time range 00:00 to 23:45
        cluster_df = cluster_df[((cluster_df['Hour'] == 0) & (cluster_df['Minute'] >= 0)) | 
                        ((cluster_df['Hour'] > 0) & (cluster_df['Hour'] < 23)) |
                        ((cluster_df['Hour'] == 23) & (cluster_df['Minute'] <= 45))]

        # Plot volume vs. time using seaborn
        sns.lineplot(data=cluster_df, x='Hour', y='VolumeCount', ax=ax)
        ax.set_title(f'Cluster {cluster_label + 1}')
        ax.set_xlabel('Time of Day (hours)')
        ax.set_ylabel('Volume')
        ax.set_xlim(0, 24)  # Set x-axis limit from 0 to 24 hours
        ax.grid(True)

    plt.tight_layout()

    pdf.savefig()
    plt.close()

def print_average_ranges(df,pdf):
    """
    Print the average range of VolumeCount, BOM_Rainfallmm, and IncidentCount for each cluster.

    Parameters:
        df (DataFrame):
            DataFrame containing the data with cluster labels.
    """
    # Group data by cluster_label
    grouped = df.groupby('cluster_label')    

    for cluster_label, cluster_data in grouped:
        # Calculate the average of VolumeCount, BOM_Rainfallmm, and IncidentCount for each cluster
        avgVolumeCount = cluster_data['VolumeCount'].mean()
        avgBOM_Rainfallmm = cluster_data['BOM_Rainfallmm'].mean()
        avgIncidentCount = cluster_data['IncidentCount'].mean()
        count = int(cluster_data['VolumeCount'].count()/96)
        ma = cluster_data['VolumeCount'].max()
        mi = cluster_data['VolumeCount'].min()
        rma = cluster_data['BOM_Rainfallmm'].max()
        rmin= cluster_data['BOM_Rainfallmm'].min()
        ima= cluster_data['IncidentCount'].max()
        imin= cluster_data['IncidentCount'].min()
        add_text_to_pdf(pdf,f"\nCluster {cluster_label}:\ncount = {count}\nAverage VolumeCount: {avgVolumeCount:.2f}\nAverage BOM_Rainfallmm: {avgBOM_Rainfallmm:.2f}\nAverage IncidentCount: {avgIncidentCount:.2f}\n\nVolumeCount range: {mi:.2f} to {ma:.2f}\nBOM_Rainfallmm range: {rmin:.2f} to {rma:.2f}\nIncidentCount range: {imin:.2f} to {ima:.2f}\n")


        #add_text_to_pdf(pdf,f"\nCluster {cluster_label}:\nVolumeCount range: {mi:.2f} to {ma:.2f}\nBOM_Rainfallmm range: {rmin:.2f} to {rma:.2f}\nIncidentCount range: {imin:.2f} to {ima:.2f}\n")


# Example usage:
# print_average_ranges(df)


def plot_cluster_volume_lt200_time(df,pdf):
    """
    Plot volume on the y-axis and time of the day on the x-axis for each cluster.

    Parameters:
        df (DataFrame):
            DataFrame containing the cluster labels and datetime information.
    """
    # Determine the number of clusters
    n_clusters = df['cluster_label'].nunique()

    # Create subplots for each cluster
    fig, axs = plt.subplots(n_clusters, figsize=(10, 6 * n_clusters), sharex=True)

    # Plot volume vs. time for each cluster
    for cluster_label, ax in zip(range(n_clusters), axs):
        cluster_df = df[df['cluster_label'] == cluster_label]

        # Extract hour and minute from LocalDateTime
        cluster_df['Hour'] = cluster_df['LocalDateTime'].dt.hour
        cluster_df['Minute'] = cluster_df['LocalDateTime'].dt.minute

        # Filter data to include only the time range 00:00 to 23:45
        cluster_df = cluster_df[(cluster_df['Hour'] == 0) & (cluster_df['Minute'] == 0) |
                                ((cluster_df['Hour'] < 23) | (cluster_df['Minute'] <= 45))]
        cluster_df = cluster_df[cluster_df['VolumeCount'].astype(int) < 200]
        
        # Plot volume vs. time
        ax.plot(cluster_df['Hour'] + cluster_df['Minute'] / 60, cluster_df['VolumeCount'])
        ax.set_title(f'Cluster {cluster_label + 1}')
        ax.set_xlabel('Time of Day (hours)')
        ax.set_ylabel('Volume')
        ax.set_xlim(0, 24)  # Set x-axis limit from 0 to 24 hours
        ax.grid(True)

    plt.tight_layout()

    pdf.savefig()
    plt.close()

def plot_cluster_public_holiday_bar(df,pdf):
    """
    Plot bar charts for each cluster, where each subplot contains two bars representing the percentage of data points
    on public holidays and non-public holidays.

    Parameters:
        df (DataFrame):
            DataFrame containing the cluster labels and public holiday information.
    """
    # Determine the number of clusters
    n_clusters = df['cluster_label'].nunique()

    # Create subplots for each cluster
    fig, axs = plt.subplots(n_clusters, figsize=(10, 6 * n_clusters))

    # Plot bar chart for each cluster
    for cluster_label, ax in zip(range(n_clusters), axs):
        cluster_df = df[df['cluster_label'] == cluster_label]

        # Calculate percentage of data points on public holidays and non-public holidays
        public_holiday_percent = (cluster_df['public_holiday'].sum() / len(cluster_df)) * 100
        non_public_holiday_percent = 100 - public_holiday_percent

        # Plot bar chart
        ax.bar(['Public Holiday', 'Non-Public Holiday'], [public_holiday_percent, non_public_holiday_percent], color=['skyblue', 'lightgreen'])
        ax.set_title(f'Cluster {cluster_label + 1}')
        ax.set_ylabel('Percentage')
        ax.set_ylim(0, 100)

        # Annotate exact percentage numbers on the chart
        for i, percentage in enumerate([public_holiday_percent, non_public_holiday_percent]):
            ax.text(i, percentage, f'{percentage:.2f}%', ha='center', va='bottom')


    plt.tight_layout()

    pdf.savefig()
    plt.close()

def plot_cluster_rainy_days_bar(df,pdf):
    """
    Plot bar charts for each cluster, where each bar represents the percentage of rainy days.

    Parameters:
        df (DataFrame):
            DataFrame containing the cluster labels and rainfall information.
    """
    # Determine the number of clusters
    n_clusters = df['cluster_label'].nunique()

    # Create subplots for each cluster
    fig, axs = plt.subplots(n_clusters, figsize=(10, 6 * n_clusters))

    # Plot bar chart for each cluster
    for cluster_label, ax in zip(range(n_clusters), axs):
        cluster_df = df[df['cluster_label'] == cluster_label]

        # Calculate percentage of rainy days
        rainy_days_percentage = (cluster_df['BOM_Rainfallmm'] > 0).mean() * 100

        # Plot bar chart
        ax.bar(['Rainy Days', 'Non-Rainy Days'], [rainy_days_percentage, 100 - rainy_days_percentage], color=['skyblue', 'orange'])
        ax.set_title(f'Cluster {cluster_label + 1}')
        ax.set_ylabel('Percentage')
        ax.set_ylim(0, 100)

        # Annotate exact percentage numbers on the chart
        for i, percentage in enumerate([rainy_days_percentage, 100 - rainy_days_percentage]):
            ax.text(i, percentage, f'{percentage:.2f}%', ha='center', va='bottom')

    plt.tight_layout()

    pdf.savefig()
    plt.close()

def plot_volume_time_for_month_day(df, pdf_pages):
    """
    Plot VolumeCount on y-axis and TimeOfDay on x-axis for each month for each day of the week.

    Parameters:
        df (DataFrame):
            DataFrame containing the data with LocalDateTime column.
        pdf_pages (PdfPages):
            PdfPages object to save the plots to PDF.
    """
    # Extract month and day from LocalDateTime column
    df['month'] = df['LocalDateTime'].dt.month
    df['day'] = df['LocalDateTime'].dt.day
    df['day_of_week'] = df['LocalDateTime'].dt.day_name()

    # Group data by month
    month_groups = df.groupby('month')

    # Iterate over each month
    for month, month_data in month_groups:
        # Group data by day of the week
        day_groups = month_data.groupby('day_of_week')
        
        # Create a new figure for the plot
        fig, axs = plt.subplots(7, figsize=(10, 15))
        
        # Iterate over each day of the week
        for i, (day, day_data) in enumerate(day_groups):
            # Plot VolumeCount vs TimeOfDay for the specified day
            axs[i].plot(day_data['LocalDateTime'].dt.hour + day_data['LocalDateTime'].dt.minute / 60, day_data['VolumeCount'])
            axs[i].set_title(f'Month {month}, {day}')
            axs[i].set_xlabel('Time of Day')
            axs[i].set_ylabel('VolumeCount')

        fig.suptitle(f'VolumeCount for Month {month}', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the plot to the PDF
        pdf_pages.savefig(fig)
        plt.close(fig)  # Close the figure to release memory

def plot_volume_time_for_day_of_week2(df, pdf_pages):
    """
    Plot VolumeCount on y-axis and TimeOfDay on x-axis for each day of the week.

    Parameters:
        df (DataFrame):
            DataFrame containing the data.
        pdf_pages (PdfPages):
            PdfPages object to save the plots to PDF.
    """
    # Extract month and day from LocalDateTime column
    df['month'] = df['LocalDateTime'].dt.month
    df['day'] = df['LocalDateTime'].dt.day
    df['day_of_week'] = df['LocalDateTime'].dt.day_name()

    # Group data by day of the week
    day_groups = df.groupby('day_of_week')

    # Map day of the week names to corresponding indexes (0: Monday, 1: Tuesday, ..., 6: Sunday)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_indexes = {day: i for i, day in enumerate(day_names)}

    # Iterate over each day of the week
    for day, day_data in day_groups:
        # Create a new figure for the subplots
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))
        fig.suptitle(f'VolumeCount for {day}', fontsize=20)
        fig.tight_layout(pad=3.0)
        axs = axs.flatten()  # Flatten the 2D array of subplots for easier iteration

        # Iterate over each month
        for i, (month, month_data) in enumerate(day_data.groupby('month')):
            # Plot VolumeCount vs TimeOfDay for the specified month and day of the week
            ax = axs[i]
            for j in range(1, 32):  # Assume maximum 31 days in a month
                # Filter data for the specified day and month
                day_month_data = month_data[month_data['day'] == j]
                if not day_month_data.empty:
                    # Extract TimeOfDay from LocalDateTime
                    #day_month_data['TimeOfDay'] = day_month_data['LocalDateTime'].dt.time
                    day_month_data['TimeOfDay'] = day_month_data['LocalDateTime'].dt.hour + day_month_data['LocalDateTime'].dt.minute / 60
                    ax.plot(day_month_data['TimeOfDay'], day_month_data['VolumeCount'], label=f'Day {j}')       
            ax.set_title(month,color='red')
            ax.set_title(month)
            ax.set_xlabel('Time of Day')
            ax.set_ylabel('VolumeCount')
            ax.legend()

        # Save the subplot to the PDF
        pdf_pages.savefig(fig)
        plt.close(fig)  # Close the figure to release memory


def generate_plots_for_mlink(mlink,df_mlink,X):
    #Print the Day of the week of the 1st observation in df_mlink
    day=(df_mlink['LocalDateTime'].dt.day_name().iloc[0])
    # Create a PDF for the Mlink
    with PdfPages(f'{day}_clustering_{mlink}_plots.pdf') as pdf:
        # Write out MLINK ID
        add_text_to_pdf(pdf, f'MLINK # {mlink} REPORT')
        #plot_rainy_vs_non_rainy_days_bar(df_mlink,pdf) 
        # Write out speed distribution
        #add_text_to_pdf(pdf, 'SPEED DISTRIBUTION PLOTS')
        #visualize_speed_distribution(df_mlink,pdf)

        # For each number of clusters from 2 to 8
        for n_clusters in range(2, 8):
            # Perform the clustering
            # Step 1: Cluster the Time Series Data
            labels = apply_kmeans(X, metric='euclidean', n_clusters=n_clusters)

            # Step 2: Expand Cluster Labels
            expanded_labels = np.repeat(labels, X.shape[1])

            #print_silhouette(df_mlink, expanded_labels,pdf)

            # Step 3: Map Cluster Labels to Original Data
            df_mlink['cluster_label'] = expanded_labels

            # Write out n clusters
            add_text_to_pdf(pdf, f'CLUSTERING WITH {n_clusters} CLUSTERS EUCLIDEAN DISTANCE METRIC')

            #day_of_week_percentages=compute_day_of_week_percentage(df_mlink)

            # Write out day of week percentages
            #add_text_to_pdf(pdf, 'DAY OF WEEK DISTRBUTION AMONG CLUSTERS')
            #plot_cluster_day_of_week_bar(df_mlink,day_of_week_percentages,pdf)
            # Write out volume vs time
            add_text_to_pdf(pdf, 'CLUSTER-WISE VOLUME VS TIME PLOTS')
            plot_cluster_volume_time2(df_mlink,pdf)
            print_average_ranges(df_mlink,pdf)
            df_mlink_200= df_mlink[df_mlink['VolumeCount'].astype(int) < 200]
            # Write out volume vs time for less than 200 volume
            #add_text_to_pdf(pdf, 'CLUSTER-WISE VOLUME < 200 VS TIME PLOTS')
            #plot_cluster_volume_lt200_time(df_mlink_200,pdf)
            # Write out public holiday bar
            add_text_to_pdf(pdf, 'CLUSTER-WISE PUBLIC HOLIDAY DISTRIBUTION')
            plot_cluster_public_holiday_bar(df_mlink,pdf)
            # Write out rainy days bar
            #add_text_to_pdf(pdf, 'CLUSTER-WISE RAINY DAYS DISTRIBUTION')
            #plot_cluster_rainy_days_bar(df_mlink,pdf)

# Read the first  file
netpres = pd.read_csv('NPR_RPT_NetPres_Fact.csv')
waze = pd.read_csv('NPR_STG_Waze_MLink_Attribs.csv')
netpres['LocalDate'] = pd.to_datetime(netpres['LocalDate'])
 
hols=pd.read_excel('WA Public Holidays 2023.xlsx')
wa=hols[hols['Jurisdiction'] == 'wa'] 
netpres['public_holiday'] = netpres['LocalDate'].isin(wa['Date']).astype(int)
#netpres['public_holiday'].value_counts() 

with open('2023 weekends.txt', 'r') as file:
    weekends = file.read().split(',')
weekends = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in weekends] 
# Convert the dates in the 'LocalDate' column to datetime objects
netpres['LocalDate'] = pd.to_datetime(netpres['LocalDate'])
# Create a binary column 'weekend' to indicate whether a given row is a weekend or not
netpres['weekend'] = netpres['LocalDate'].dt.date.isin([date.date() for date in weekends]).astype(int)

import pandas as pd

def add_incident_count(df, df2):
    """
    Add an IncidentCount column to the original DataFrame based on incident occurrences in df2.

    Parameters:
        df (DataFrame):
            Original DataFrame containing the LocalDateTime column.
        df2 (DataFrame):
            DataFrame containing the incident information and TimeOfTheIncident column.

    Returns:
        DataFrame:
            Modified DataFrame with the IncidentCount column added.
    """
    # Convert LocalDateTime and TimeOfTheIncident to datetime
    df['LocalDate'] = pd.to_datetime(df['LocalDate'])
    df2['FirstDateTimeLocal'] = pd.to_datetime(df2['FirstDateTimeLocal'])

    # Group incidents in df2 by date and count occurrences
    incident_counts = df2.groupby(df2['FirstDateTimeLocal'].dt.date)['AlertType'].count().reset_index()
    incident_counts.columns = ['Date', 'IncidentCount']

    # Merge incident counts with df
    df['Date'] = df['LocalDate'].dt.date
    df = pd.merge(df, incident_counts, on='Date', how='left')

    # Fill NaN values in IncidentCount column with 0
    df['IncidentCount'].fillna(0, inplace=True)

    # Drop the temporary 'Date' column
    df.drop(columns=['Date'], inplace=True)

    return df

def filter_mlink(mlink,df,waze):
    df2=df[['MLinkID','LocalDate','LocalTime','SpeedKmh','VolumeCount','BOM_Rainfallmm','public_holiday','weekend']]
    df2=df.loc[df['MLinkID']==mlink]
    df2=df2.drop(['MLinkID'],axis=1)
    df2 = df2[(df2['LocalDate'].dt.dayofweek == 0)]  # Monday is represented by 0
    df2 = add_incident_count(df2, waze[waze['MLinkId'] == 721])
    df2['LocalDateTime'] = pd.to_datetime(df2['LocalDate'].astype(str) + ' ' + df2['LocalTime'].astype(str))
    df2=df2.drop(['LocalDate','LocalTime'],axis=1)
    df2=df2[['LocalDateTime',
             'VolumeCount',
             'SpeedKmh', 
     'BOM_Rainfallmm',
     'public_holiday',
     'weekend', 'IncidentCount']]
    df2['BOM_Rainfallmm'] = df2['BOM_Rainfallmm'].fillna(0)
    df2 = df2.sort_values(by='LocalDateTime')
    df2=df2.reset_index(drop=True)
    
    # Group data by desired factors and convert to time series
    groups = df2.groupby(df2['LocalDateTime'].dt.date)
    time_series = []
    for group_name, group_data in groups:
        time_series.append(group_data[['VolumeCount']].values)

    # Convert to 3D NumPy array
    X = to_time_series(time_series)
    #print(X.shape)  # Output should be (number_of_time_series, length_of_each_time_series, number_of_features)
    return df2,X

# Get the unique Mlinks
#mlinks = netpres['MLinkID'].unique()
mlinks=[721]

# Example usage:


# Generate plots for each Mlink
for mlink in mlinks:
    mlinkdf,filtered_mlink_3d = filter_mlink(mlink,netpres,waze)
    with PdfPages('x.pdf') as pdf:
        #plot_volume_time_for_month_day(mlinkdf, pdf)
        #plot_volume_time_for_day_of_week2(mlinkdf, pdf)        
        generate_plots_for_mlink(mlink,mlinkdf,filtered_mlink_3d)