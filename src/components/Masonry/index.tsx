import React from 'react';
import clsx from 'clsx';
import MasonryItem from './MasonryItem';

import styles from './styles.module.css';

export default function Masonry({ items }: { items: MasonryItem[] }) {

    return (
        <div className={clsx(styles.masonry)}>
            {
                items.map((item, index) => {
                    return (
                        <div className={clsx(styles.item)} key={index}>
                            <>
                                <img src={item.image} alt={item.title} loading='lazy' decoding='async'/>
                                <p>{item.description}</p>
                            </>
                        </div>
                    )
                })
            }
        </div>
    );
}