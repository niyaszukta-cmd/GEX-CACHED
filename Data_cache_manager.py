"""
NYZTrade Historical Data Cache Manager
Stores and retrieves up to 5 years of historical GEX/DEX data
"""

import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
from typing import Optional, Dict, List
import hashlib

class DataCacheManager:
    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize the cache manager
        
        Args:
            cache_dir: Directory to store cached data files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.pkl"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _generate_cache_key(self, symbol: str, date: str, interval: str, 
                           expiry_code: int, expiry_flag: str, strikes: List[str]) -> str:
        """Generate unique cache key for data combination"""
        strikes_str = ",".join(sorted(strikes))
        key_string = f"{symbol}_{date}_{interval}_{expiry_code}_{expiry_flag}_{strikes_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_filepath(self, cache_key: str) -> Path:
        """Get filepath for cached data"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def is_cached(self, symbol: str, date: str, interval: str, 
                  expiry_code: int, expiry_flag: str, strikes: List[str]) -> bool:
        """Check if data exists in cache"""
        cache_key = self._generate_cache_key(symbol, date, interval, expiry_code, expiry_flag, strikes)
        return cache_key in self.metadata and self._get_cache_filepath(cache_key).exists()
    
    def save_to_cache(self, df: pd.DataFrame, meta: Dict, symbol: str, date: str, 
                     interval: str, expiry_code: int, expiry_flag: str, strikes: List[str]):
        """
        Save fetched data to cache
        
        Args:
            df: DataFrame with historical data
            meta: Metadata dictionary
            symbol, date, interval, expiry_code, expiry_flag, strikes: Query parameters
        """
        cache_key = self._generate_cache_key(symbol, date, interval, expiry_code, expiry_flag, strikes)
        cache_filepath = self._get_cache_filepath(cache_key)
        
        # Save data
        cache_data = {
            'df': df,
            'meta': meta,
            'cached_at': datetime.now().isoformat(),
            'symbol': symbol,
            'date': date,
            'interval': interval,
            'expiry_code': expiry_code,
            'expiry_flag': expiry_flag,
            'strikes': strikes
        }
        
        with open(cache_filepath, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Update metadata
        self.metadata[cache_key] = {
            'symbol': symbol,
            'date': date,
            'interval': interval,
            'expiry_code': expiry_code,
            'expiry_flag': expiry_flag,
            'strikes': strikes,
            'cached_at': datetime.now().isoformat(),
            'file_size': cache_filepath.stat().st_size
        }
        
        self._save_metadata()
    
    def load_from_cache(self, symbol: str, date: str, interval: str, 
                       expiry_code: int, expiry_flag: str, strikes: List[str]) -> Optional[tuple]:
        """
        Load data from cache
        
        Returns:
            tuple: (df, meta) if found, None otherwise
        """
        cache_key = self._generate_cache_key(symbol, date, interval, expiry_code, expiry_flag, strikes)
        cache_filepath = self._get_cache_filepath(cache_key)
        
        if not cache_filepath.exists():
            return None
        
        try:
            with open(cache_filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            return cache_data['df'], cache_data['meta']
        except Exception as e:
            st.warning(f"Failed to load cached data: {str(e)}")
            return None
    
    def get_cached_dates(self, symbol: str = None) -> List[str]:
        """Get list of all cached dates, optionally filtered by symbol"""
        cached_dates = set()
        
        for cache_key, info in self.metadata.items():
            if symbol is None or info['symbol'] == symbol:
                cached_dates.add(info['date'])
        
        return sorted(list(cached_dates), reverse=True)
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_entries = len(self.metadata)
        total_size = sum(info.get('file_size', 0) for info in self.metadata.values())
        
        symbols = set(info['symbol'] for info in self.metadata.values())
        dates = set(info['date'] for info in self.metadata.values())
        
        oldest_date = min(dates) if dates else None
        newest_date = max(dates) if dates else None
        
        return {
            'total_entries': total_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'symbols': sorted(list(symbols)),
            'unique_dates': len(dates),
            'oldest_date': oldest_date,
            'newest_date': newest_date,
            'date_range': dates
        }
    
    def clear_cache(self, symbol: str = None, before_date: str = None):
        """
        Clear cache entries
        
        Args:
            symbol: If provided, only clear this symbol
            before_date: If provided, only clear dates before this
        """
        keys_to_remove = []
        
        for cache_key, info in self.metadata.items():
            should_remove = True
            
            if symbol and info['symbol'] != symbol:
                should_remove = False
            
            if before_date and info['date'] >= before_date:
                should_remove = False
            
            if should_remove:
                keys_to_remove.append(cache_key)
        
        # Remove files and metadata
        for cache_key in keys_to_remove:
            cache_filepath = self._get_cache_filepath(cache_key)
            if cache_filepath.exists():
                cache_filepath.unlink()
            
            del self.metadata[cache_key]
        
        self._save_metadata()
        
        return len(keys_to_remove)
    
    def import_csv(self, csv_filepath: str, symbol: str, date: str, interval: str,
                   expiry_code: int, expiry_flag: str, strikes: List[str]):
        """
        Import data from CSV file into cache
        
        Args:
            csv_filepath: Path to CSV file
            symbol, date, interval, expiry_code, expiry_flag, strikes: Data parameters
        """
        df = pd.read_csv(csv_filepath)
        
        # Convert timestamp column if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create metadata
        meta = {
            'symbol': symbol,
            'date': date,
            'spot_price': df['spot_price'].iloc[-1] if len(df) > 0 else 0,
            'total_records': len(df),
            'time_range': f"{df['time'].min()} - {df['time'].max()}" if 'time' in df.columns else "Unknown",
            'strikes_count': df['strike'].nunique() if 'strike' in df.columns else 0,
            'interval': interval,
            'expiry_code': expiry_code,
            'expiry_flag': expiry_flag,
            'imported_from_csv': csv_filepath
        }
        
        self.save_to_cache(df, meta, symbol, date, interval, expiry_code, expiry_flag, strikes)
    
    def export_cache_to_csv(self, cache_key: str, output_dir: str = "exports"):
        """Export cached data to CSV"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        cache_filepath = self._get_cache_filepath(cache_key)
        
        if not cache_filepath.exists():
            return None
        
        with open(cache_filepath, 'rb') as f:
            cache_data = pickle.load(f)
        
        df = cache_data['df']
        info = self.metadata.get(cache_key, {})
        
        filename = f"{info.get('symbol', 'unknown')}_{info.get('date', 'unknown')}.csv"
        output_filepath = output_path / filename
        
        df.to_csv(output_filepath, index=False)
        
        return str(output_filepath)
    
    def get_available_configurations(self, symbol: str, date: str) -> List[Dict]:
        """Get all available cached configurations for a symbol and date"""
        configs = []
        
        for cache_key, info in self.metadata.items():
            if info['symbol'] == symbol and info['date'] == date:
                configs.append({
                    'interval': info['interval'],
                    'expiry_code': info['expiry_code'],
                    'expiry_flag': info['expiry_flag'],
                    'strikes': info['strikes'],
                    'cache_key': cache_key,
                    'cached_at': info['cached_at']
                })
        
        return configs
    
    def batch_download_dates(self, symbol: str, start_date: str, end_date: str,
                            fetcher, strikes: List[str], interval: str = "60",
                            expiry_code: int = 1, expiry_flag: str = "WEEK"):
        """
        Batch download historical data for a date range
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fetcher: DhanHistoricalFetcher instance
            strikes: List of strikes to fetch
            interval: Data interval
            expiry_code: Expiry code
            expiry_flag: Expiry flag
            
        Returns:
            dict: Statistics about download
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates_to_fetch = []
        current = start
        
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:  # Monday=0, Friday=4
                date_str = current.strftime('%Y-%m-%d')
                
                # Check if already cached
                if not self.is_cached(symbol, date_str, interval, expiry_code, expiry_flag, strikes):
                    dates_to_fetch.append(date_str)
            
            current += timedelta(days=1)
        
        stats = {
            'total_dates': len(dates_to_fetch),
            'successful': 0,
            'failed': 0,
            'skipped_cached': 0
        }
        
        if len(dates_to_fetch) == 0:
            st.info("All dates already cached!")
            return stats
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, date_str in enumerate(dates_to_fetch):
            status_text.text(f"Downloading {date_str}... ({i+1}/{len(dates_to_fetch)})")
            
            try:
                df, meta = fetcher.process_historical_data(
                    symbol, date_str, strikes, interval, expiry_code, expiry_flag
                )
                
                if df is not None and len(df) > 0:
                    self.save_to_cache(df, meta, symbol, date_str, interval, 
                                     expiry_code, expiry_flag, strikes)
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
            
            except Exception as e:
                st.warning(f"Failed to fetch {date_str}: {str(e)}")
                stats['failed'] += 1
            
            progress_bar.progress((i + 1) / len(dates_to_fetch))
        
        progress_bar.empty()
        status_text.empty()
        
        return stats
