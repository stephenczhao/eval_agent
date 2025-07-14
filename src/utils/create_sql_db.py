import sqlite3
import pandas as pd
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_tennis_database(db_path='tennis_matches.db'):
    """
    Create a SQLite database optimized for LLM Text2SQL queries from ATP and WTA tennis data.
    
    Schema Design Principles:
    - Separate players table with unique IDs for better normalization
    - Unified match table structure for both ATP and WTA
    - Descriptive column names for better LLM understanding
    - Proper data types for numeric operations and date queries
    - Indexes on commonly queried columns
    - Normalized tournament and surface values
    """
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS tennis_matches")
    cursor.execute("DROP TABLE IF EXISTS players")
    cursor.execute("DROP VIEW IF EXISTS player_stats")
    
    # Create the players table first
    create_players_table_sql = """
    CREATE TABLE players (
        player_id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT NOT NULL UNIQUE,
        normalized_name TEXT NOT NULL,
        tour_type TEXT NOT NULL CHECK (tour_type IN ('ATP', 'WTA', 'BOTH')),
        first_appearance_date DATE,
        last_appearance_date DATE,
        total_matches INTEGER DEFAULT 0,
        total_wins INTEGER DEFAULT 0,
        
        -- Will be updated as we process matches
        best_ranking INTEGER,
        highest_points INTEGER
    )
    """
    
    cursor.execute(create_players_table_sql)
    
    # Create the main matches table with foreign keys to players
    create_table_sql = """
    CREATE TABLE tennis_matches (
        -- Primary key and tour identification
        match_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tour_type TEXT NOT NULL CHECK (tour_type IN ('ATP', 'WTA')),
        
        -- Tournament information
        tournament_location TEXT NOT NULL,
        tournament_name TEXT NOT NULL,
        match_date DATE NOT NULL,
        
        -- Court and surface details
        court_type TEXT CHECK (court_type IN ('Indoor', 'Outdoor')),
        surface_type TEXT CHECK (surface_type IN ('Hard', 'Clay', 'Grass', 'Carpet')),
        
        -- Match format and round
        tournament_round TEXT NOT NULL,
        best_of_sets INTEGER CHECK (best_of_sets IN (3, 5)),
        
        -- Player information (foreign keys)
        winner_id INTEGER NOT NULL,
        loser_id INTEGER NOT NULL,
        
        -- Keep names for easy querying (denormalized for LLM convenience)
        winner_name TEXT NOT NULL,
        loser_name TEXT NOT NULL,
        
        -- Rankings (lower number = better rank, NULL for unranked)
        winner_rank INTEGER,
        loser_rank INTEGER,
        
        -- Points (official ATP/WTA points)
        winner_points INTEGER,
        loser_points INTEGER,
        
        -- Derived fields for easier querying
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        tournament_level TEXT, -- Will be derived from tournament name patterns
        ranking_difference INTEGER, -- winner_rank - loser_rank (negative = higher ranked beat lower ranked)
        points_difference INTEGER,  -- winner_points - loser_points
        
        -- Foreign key constraints
        FOREIGN KEY (winner_id) REFERENCES players(player_id),
        FOREIGN KEY (loser_id) REFERENCES players(player_id)
    )
    """
    
    cursor.execute(create_table_sql)
    
    # Create indexes for common query patterns
    indexes = [
        # Players table indexes
        "CREATE INDEX idx_players_name ON players(player_name)",
        "CREATE INDEX idx_players_normalized ON players(normalized_name)",
        "CREATE INDEX idx_players_tour ON players(tour_type)",
        
        # Matches table indexes
        "CREATE INDEX idx_tour_type ON tennis_matches(tour_type)",
        "CREATE INDEX idx_year ON tennis_matches(year)",
        "CREATE INDEX idx_tournament ON tennis_matches(tournament_name)",
        "CREATE INDEX idx_surface ON tennis_matches(surface_type)",
        "CREATE INDEX idx_round ON tennis_matches(tournament_round)",
        "CREATE INDEX idx_winner_id ON tennis_matches(winner_id)",
        "CREATE INDEX idx_loser_id ON tennis_matches(loser_id)",
        "CREATE INDEX idx_winner_name ON tennis_matches(winner_name)",
        "CREATE INDEX idx_loser_name ON tennis_matches(loser_name)",
        "CREATE INDEX idx_date ON tennis_matches(match_date)",
        "CREATE INDEX idx_location ON tennis_matches(tournament_location)",
        "CREATE INDEX idx_winner_rank ON tennis_matches(winner_rank)",
        "CREATE INDEX idx_loser_rank ON tennis_matches(loser_rank)",
        "CREATE INDEX idx_composite_player_year ON tennis_matches(winner_id, year)",
        "CREATE INDEX idx_composite_tournament_year ON tennis_matches(tournament_name, year)"
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)
    
    logger.info("Database schema created successfully")
    
    # Function to normalize player names
    def normalize_player_name(name):
        """Normalize player names for consistency"""
        if not name:
            return name
        
        # Basic normalization
        name = str(name).strip()
        
        # Remove extra spaces
        name = ' '.join(name.split())
        
        # Handle common abbreviations and formatting
        name = name.replace('.', '')  # Remove periods
        
        return name
    
    # Function to get or create player
    def get_or_create_player(player_name, tour_type, cursor):
        """Get existing player ID or create new player"""
        if not player_name:
            return None
            
        normalized_name = normalize_player_name(player_name)
        
        # Try to find existing player
        cursor.execute("""
            SELECT player_id FROM players 
            WHERE normalized_name = ? OR player_name = ?
        """, (normalized_name, player_name))
        
        result = cursor.fetchone()
        if result:
            # Update tour_type if player plays both tours
            cursor.execute("""
                UPDATE players 
                SET tour_type = CASE 
                    WHEN tour_type != ? AND tour_type != 'BOTH' THEN 'BOTH'
                    ELSE tour_type
                END
                WHERE player_id = ?
            """, (tour_type, result[0]))
            return result[0]
        
        # Create new player
        cursor.execute("""
            INSERT INTO players (player_name, normalized_name, tour_type)
            VALUES (?, ?, ?)
        """, (player_name, normalized_name, tour_type))
        
        return cursor.lastrowid
    
    # Function to normalize tournament round names
    def normalize_round(round_name):
        """Normalize round names for consistency"""
        if not round_name:
            return round_name
        
        round_name = str(round_name).strip()
        
        # Common normalizations
        round_mappings = {
            'The Final': 'Final',
            '1st Round': 'Round 1',
            '2nd Round': 'Round 2', 
            '3rd Round': 'Round 3',
            'Quarterfinals': 'Quarterfinal',
            'Semifinals': 'Semifinal'
        }
        
        return round_mappings.get(round_name, round_name)
    
    # Function to determine tournament level
    def get_tournament_level(tournament_name, tour_type):
        """Determine tournament level from name patterns"""
        if not tournament_name:
            return None
            
        name_lower = tournament_name.lower()
        
        if tour_type == 'ATP':
            if 'grand slam' in name_lower or any(slam in name_lower for slam in ['wimbledon', 'us open', 'french open', 'australian open']):
                return 'Grand Slam'
            elif 'masters' in name_lower or 'atp 1000' in name_lower:
                return 'ATP 1000'
            elif 'atp 500' in name_lower:
                return 'ATP 500'
            elif 'atp 250' in name_lower:
                return 'ATP 250'
            else:
                return 'Other'
        else:  # WTA
            if 'grand slam' in name_lower or any(slam in name_lower for slam in ['wimbledon', 'us open', 'french open', 'australian open']):
                return 'Grand Slam'
            elif 'wta 1000' in name_lower:
                return 'WTA 1000'
            elif 'wta 500' in name_lower:
                return 'WTA 500'
            elif 'wta 250' in name_lower:
                return 'WTA 250'
            else:
                return 'Other'
    
    # Load and process ATP data
    logger.info("Loading ATP data...")
    atp_file = Path('tennis_data/data/atp_men/atp_2023-2025.csv')
    if atp_file.exists():
        atp_df = pd.read_csv(atp_file)
        
        # Process ATP data
        for _, row in atp_df.iterrows():
            try:
                # Parse date and convert to ISO format string to avoid deprecation warnings
                date_obj = pd.to_datetime(row['Date']).date()
                match_date = date_obj.isoformat()  # Convert to ISO format string (YYYY-MM-DD)
                year = date_obj.year
                month = date_obj.month
                
                # Calculate derived fields
                winner_rank = row['WRank'] if pd.notna(row['WRank']) else None
                loser_rank = row['LRank'] if pd.notna(row['LRank']) else None
                winner_points = row['WPts'] if pd.notna(row['WPts']) else None
                loser_points = row['LPts'] if pd.notna(row['LPts']) else None
                
                ranking_diff = None
                if winner_rank is not None and loser_rank is not None:
                    ranking_diff = winner_rank - loser_rank
                
                points_diff = None
                if winner_points is not None and loser_points is not None:
                    points_diff = winner_points - loser_points
                
                tournament_level = get_tournament_level(row['Tournament'], 'ATP')
                normalized_round = normalize_round(row['Round'])
                
                # Get or create player IDs
                winner_id = get_or_create_player(row['Winner'], 'ATP', cursor)
                loser_id = get_or_create_player(row['Loser'], 'ATP', cursor)
                
                if winner_id is None or loser_id is None:
                    logger.warning(f"Could not create player IDs for match: {row['Winner']} vs {row['Loser']}")
                    continue
                
                # Insert into database
                cursor.execute("""
                    INSERT INTO tennis_matches (
                        tour_type, tournament_location, tournament_name, match_date,
                        court_type, surface_type, tournament_round, best_of_sets,
                        winner_id, loser_id, winner_name, loser_name, winner_rank, loser_rank,
                        winner_points, loser_points, year, month, tournament_level,
                        ranking_difference, points_difference
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'ATP', row['Location'], row['Tournament'], match_date,
                    row['Court'], row['Surface'], normalized_round, row['Best of'],
                    winner_id, loser_id, row['Winner'], row['Loser'], winner_rank, loser_rank,
                    winner_points, loser_points, year, month, tournament_level,
                    ranking_diff, points_diff
                ))
            except Exception as e:
                logger.warning(f"Error processing ATP row: {e}")
                continue
        
        logger.info(f"Processed {len(atp_df)} ATP matches")
    else:
        logger.warning("ATP data file not found")
    
    # Load and process WTA data
    logger.info("Loading WTA data...")
    wta_file = Path('tennis_data/data/wta_women/wta_2023-2025.csv')
    if wta_file.exists():
        wta_df = pd.read_csv(wta_file)
        
        # Process WTA data
        for _, row in wta_df.iterrows():
            try:
                # Parse date and convert to ISO format string to avoid deprecation warnings
                date_obj = pd.to_datetime(row['Date']).date()
                match_date = date_obj.isoformat()  # Convert to ISO format string (YYYY-MM-DD)
                year = date_obj.year
                month = date_obj.month
                
                # Calculate derived fields
                winner_rank = row['WRank'] if pd.notna(row['WRank']) else None
                loser_rank = row['LRank'] if pd.notna(row['LRank']) else None
                winner_points = row['WPts'] if pd.notna(row['WPts']) else None
                loser_points = row['LPts'] if pd.notna(row['LPts']) else None
                
                ranking_diff = None
                if winner_rank is not None and loser_rank is not None:
                    ranking_diff = winner_rank - loser_rank
                
                points_diff = None
                if winner_points is not None and loser_points is not None:
                    points_diff = winner_points - loser_points
                
                tournament_level = get_tournament_level(row['Tournament'], 'WTA')
                normalized_round = normalize_round(row['Round'])
                
                # Get or create player IDs
                winner_id = get_or_create_player(row['Winner'], 'WTA', cursor)
                loser_id = get_or_create_player(row['Loser'], 'WTA', cursor)
                
                if winner_id is None or loser_id is None:
                    logger.warning(f"Could not create player IDs for match: {row['Winner']} vs {row['Loser']}")
                    continue
                
                # Insert into database
                cursor.execute("""
                    INSERT INTO tennis_matches (
                        tour_type, tournament_location, tournament_name, match_date,
                        court_type, surface_type, tournament_round, best_of_sets,
                        winner_id, loser_id, winner_name, loser_name, winner_rank, loser_rank,
                        winner_points, loser_points, year, month, tournament_level,
                        ranking_difference, points_difference
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'WTA', row['Location'], row['Tournament'], match_date,
                    row['Court'], row['Surface'], normalized_round, row['Best of'],
                    winner_id, loser_id, row['Winner'], row['Loser'], winner_rank, loser_rank,
                    winner_points, loser_points, year, month, tournament_level,
                    ranking_diff, points_diff
                ))
            except Exception as e:
                logger.warning(f"Error processing WTA row: {e}")
                continue
        
        logger.info(f"Processed {len(wta_df)} WTA matches")
    else:
        logger.warning("WTA data file not found")
    
    # Clean up data outliers
    logger.info("Cleaning up data outliers...")
    
    # Remove outlier matches (e.g., 2015 date in 2023-2025 dataset)
    cursor.execute("DELETE FROM tennis_matches WHERE year < 2023")
    outliers_removed = cursor.rowcount
    if outliers_removed > 0:
        logger.info(f"Removed {outliers_removed} outlier matches with dates before 2023")
    
    # Update player statistics
    logger.info("Updating player statistics...")
    
    # Update player match counts and date ranges
    cursor.execute("""
        UPDATE players SET 
            total_matches = (
                SELECT COUNT(*) FROM tennis_matches 
                WHERE winner_id = players.player_id OR loser_id = players.player_id
            ),
            total_wins = (
                SELECT COUNT(*) FROM tennis_matches 
                WHERE winner_id = players.player_id
            ),
            first_appearance_date = (
                SELECT MIN(match_date) FROM tennis_matches 
                WHERE winner_id = players.player_id OR loser_id = players.player_id
            ),
            last_appearance_date = (
                SELECT MAX(match_date) FROM tennis_matches 
                WHERE winner_id = players.player_id OR loser_id = players.player_id
            ),
            best_ranking = (
                SELECT MIN(CASE 
                    WHEN winner_id = players.player_id AND winner_rank IS NOT NULL THEN winner_rank
                    WHEN loser_id = players.player_id AND loser_rank IS NOT NULL THEN loser_rank
                END) FROM tennis_matches 
                WHERE winner_id = players.player_id OR loser_id = players.player_id
            ),
            highest_points = (
                SELECT MAX(CASE 
                    WHEN winner_id = players.player_id AND winner_points IS NOT NULL THEN winner_points
                    WHEN loser_id = players.player_id AND loser_points IS NOT NULL THEN loser_points
                END) FROM tennis_matches 
                WHERE winner_id = players.player_id OR loser_id = players.player_id
            )
    """)
    
    # Commit changes and create summary statistics
    conn.commit()
    
    # Generate summary statistics
    cursor.execute("SELECT COUNT(*) FROM tennis_matches")
    total_matches = cursor.fetchone()[0]
    
    cursor.execute("SELECT tour_type, COUNT(*) FROM tennis_matches GROUP BY tour_type")
    tour_counts = cursor.fetchall()
    
    cursor.execute("SELECT MIN(match_date), MAX(match_date) FROM tennis_matches")
    date_range = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(DISTINCT tournament_name) FROM tennis_matches")
    unique_tournaments = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM players")
    unique_players = cursor.fetchone()[0]
    
    cursor.execute("SELECT tour_type, COUNT(*) FROM players GROUP BY tour_type")
    player_tour_counts = cursor.fetchall()
    
    logger.info("Database creation completed!")
    logger.info(f"Total matches: {total_matches}")
    logger.info(f"Tour distribution: {dict(tour_counts)}")
    logger.info(f"Date range: {date_range[0]} to {date_range[1]}")
    logger.info(f"Unique tournaments: {unique_tournaments}")
    logger.info(f"Unique players: {unique_players}")
    logger.info(f"Player tour distribution: {dict(player_tour_counts)}")
    
    # Create enhanced views for common queries
    cursor.execute("""
        CREATE VIEW player_match_stats AS
        SELECT 
            p.player_id,
            p.player_name,
            p.tour_type,
            p.total_matches,
            p.total_wins,
            ROUND(p.total_wins * 100.0 / p.total_matches, 2) as win_percentage,
            p.best_ranking,
            p.highest_points,
            p.first_appearance_date,
            p.last_appearance_date
        FROM players p
        WHERE p.total_matches > 0
    """)
    
    cursor.execute("""
        CREATE VIEW head_to_head AS
        SELECT 
            p1.player_name as player1,
            p2.player_name as player2,
            COUNT(*) as h2h_matches,
            SUM(CASE WHEN m.winner_id = p1.player_id THEN 1 ELSE 0 END) as player1_wins,
            SUM(CASE WHEN m.winner_id = p2.player_id THEN 1 ELSE 0 END) as player2_wins
        FROM tennis_matches m
        JOIN players p1 ON (m.winner_id = p1.player_id OR m.loser_id = p1.player_id)
        JOIN players p2 ON (m.winner_id = p2.player_id OR m.loser_id = p2.player_id)
        WHERE p1.player_id < p2.player_id 
        AND ((m.winner_id = p1.player_id AND m.loser_id = p2.player_id) OR 
             (m.winner_id = p2.player_id AND m.loser_id = p1.player_id))
        GROUP BY p1.player_id, p2.player_id
        HAVING h2h_matches > 1
    """)
    
    cursor.execute("""
        CREATE VIEW surface_performance AS
        SELECT 
            p.player_name,
            m.surface_type,
            COUNT(*) as matches_played,
            SUM(CASE WHEN m.winner_id = p.player_id THEN 1 ELSE 0 END) as wins,
            ROUND(SUM(CASE WHEN m.winner_id = p.player_id THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_percentage
        FROM players p
        JOIN tennis_matches m ON (m.winner_id = p.player_id OR m.loser_id = p.player_id)
        WHERE m.surface_type IS NOT NULL
        GROUP BY p.player_id, m.surface_type
        HAVING matches_played >= 5
    """)
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database saved as {db_path}")
    return db_path

if __name__ == "__main__":

    # Preprocess and merge excel data
    atp_path = "tennis_data/data/atp_men/"
    wta_path = "tennis_data/data/wta_women/"

    atp_2023 = pd.read_excel(atp_path + "2023.xlsx")
    wta_2023 = pd.read_excel(wta_path + "2023.xlsx")
    print("atp_2023: ", atp_2023.shape)
    print("wta_2023: ", wta_2023.shape, "\n")

    atp_2024 = pd.read_excel(atp_path + "2024.xlsx")
    wta_2024 = pd.read_excel(wta_path + "2024.xlsx")
    print("atp_2024: ", atp_2024.shape)
    print("wta_2024: ", wta_2024.shape, "\n")

    atp_2025 = pd.read_excel(atp_path + "2025.xlsx")
    wta_2025 = pd.read_excel(wta_path + "2025.xlsx")
    print("atp_2025: ", atp_2025.shape)
    print("wta_2025: ", wta_2025.shape, "\n")

    # merge atp and wta data
    atp_all = pd.concat([atp_2023, atp_2024, atp_2025])
    wta_all = pd.concat([wta_2023, wta_2024, wta_2025])

    # save merged data
    atp_all.to_csv("tennis_data/data/atp_men/atp_2023-2025.csv", index=False)
    wta_all.to_csv("tennis_data/data/wta_women/wta_2023-2025.csv", index=False)

    print("ATP columns: ", atp_all.columns)
    print("WTA columns: ", wta_all.columns)

    # Check if database already exists and ask user for confirmation
    db_path = 'tennis_data/tennis_matches.db'
    if Path(db_path).exists():
        print(f"\nDatabase '{db_path}' already exists.")
        user_input = input("Do you want to reinitialize it? This will delete the existing database. (Y/N): ").strip().upper()
        
        if user_input in ['Y', 'YES']:
            print(f"Deleting existing database '{db_path}'...")
            Path(db_path).unlink()
            print("Database deleted successfully.")
        elif user_input in ['N', 'NO']:
            print("Script stopped. Existing database preserved.")
            sys.exit(0)
        else:
            print("Invalid input. Please enter Y or N. Script stopped.")
            sys.exit(1)

    # Create the database
    db_path = create_tennis_database(db_path)
    
    # Test the database with some sample queries
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n=== Sample Queries for Testing Enhanced Schema ===")
    
    # Query 1: Count matches by surface
    print("\n1. Matches by surface:")
    cursor.execute("SELECT surface_type, COUNT(*) FROM tennis_matches GROUP BY surface_type ORDER BY COUNT(*) DESC")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")
    
    # Query 2: Top 5 most active players (using players table)
    print("\n2. Top 5 most active players:")
    cursor.execute("""
        SELECT player_name, total_matches, total_wins, win_percentage
        FROM player_match_stats 
        ORDER BY total_matches DESC 
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} matches, {row[2]} wins ({row[3]}%)")
    
    # Query 3: Players who compete in both ATP and WTA
    print("\n3. Players competing in both tours:")
    cursor.execute("SELECT player_name, total_matches FROM players WHERE tour_type = 'BOTH'")
    both_tour_players = cursor.fetchall()
    if both_tour_players:
        for row in both_tour_players:
            print(f"  {row[0]}: {row[1]} total matches")
    else:
        print("  No players found competing in both tours")
    
    # Query 4: Head-to-head matchups (sample)
    print("\n4. Sample head-to-head records:")
    cursor.execute("SELECT * FROM head_to_head ORDER BY h2h_matches DESC LIMIT 3")
    for row in cursor.fetchall():
        print(f"  {row[0]} vs {row[1]}: {row[2]} matches ({row[3]}-{row[4]})")
    
    # Query 5: Surface specialists
    print("\n5. Best surface win percentages (min 10 matches):")
    cursor.execute("""
        SELECT player_name, surface_type, matches_played, win_percentage
        FROM surface_performance 
        WHERE matches_played >= 10
        ORDER BY win_percentage DESC 
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]} on {row[1]}: {row[3]}% ({row[2]} matches)")
    
    # Query 6: Demonstrate player ID queries
    print("\n6. Sample player lookup by ID:")
    cursor.execute("""
        SELECT m.match_date, m.tournament_name, w.player_name as winner, l.player_name as loser
        FROM tennis_matches m
        JOIN players w ON m.winner_id = w.player_id
        JOIN players l ON m.loser_id = l.player_id
        ORDER BY m.match_date DESC
        LIMIT 3
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[2]} def. {row[3]} at {row[1]}")
    
    conn.close()
    print(f"\nDatabase successfully created at: {db_path}")
    print("\nKey Benefits of Enhanced Schema:")
    print("- Players table with unique IDs enables better player queries")
    print("- Foreign key relationships improve data integrity") 
    print("- Pre-computed player statistics for faster queries")
    print("- Enhanced views for head-to-head and surface analysis")
    print("- Better support for LLM Text2SQL queries about specific players")